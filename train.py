import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import MeituData
from metrics import evaluate
from model import SPCDH, adaptive_weight
ROOT = Path(__file__).resolve().parent

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def package_fingerprint():
    sha = hashlib.sha256()
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'core').glob('*.py'))
    for path in sorted(paths):
        sha.update(path.relative_to(ROOT).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()

@contextmanager
def run_lock(path):
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])

def train_epoch(model, loader, optimizer, device, epoch, max_steps=None):
    model.train()
    sums, count = ({}, 0)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        total, parts = model.compute_loss(*(x.to(device, non_blocking=True) for x in batch), epoch=epoch)
        if not torch.isfinite(total):
            raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {count}')
        total.backward()
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
        if not finite:
            raise FloatingPointError(f'Non-finite gradients at epoch {epoch}, batch {count}')
        optimizer.step()
        for name, value in dict(parts, loss_total=total).items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        count += 1
        if max_steps is not None and count >= max_steps:
            break
    if not count:
        raise ValueError('Empty training loader')
    return {name: value / count for name, value in sums.items()}

def run(args, output):
    model_params = json.loads(args.model_params.read_text(encoding='utf-8'))
    if not isinstance(model_params, dict) or set(model_params) != {'fusion_params', 'hash_params', 'metric_params'}:
        raise ValueError('Model parameters require fusion_params, hash_params and metric_params')
    model_parameters_sha256 = hashlib.sha256(json.dumps(model_params, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA PyTorch and a visible GPU are required; check the server environment')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    identity = {'model_parameters_sha256': model_parameters_sha256, 'package_sha256': package_fingerprint(), 'bits': args.bits, 'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed, 'smoke_test': args.smoke_test, 'checkpoint_selection': 'fixed_final_epoch'}
    result_path = output / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous.get('identity') != identity:
            raise ValueError('Existing result has a different protocol; choose a new output root')
        print(json.dumps({'status': 'already_complete', **previous}, ensure_ascii=False), flush=True)
        return
    print(f'Loading existing Meitu features and point clouds, {args.bits} bit', flush=True)
    data = MeituData(args.data_root)
    if len(data.classes) != 63 or {k: len(v[0]) for k, v in data.splits.items()} != {'train': 16000, 'val': 2000, 'test': 2000}:
        raise ValueError('Meitu classes/splits differ from the fixed protocol')
    train_set = data.dataset('train', 256 if args.smoke_test else None)
    query_set = data.dataset('test', 32 if args.smoke_test else None)
    db_set = data.dataset('val', 128 if args.smoke_test else None)
    loader_options = dict(num_workers=0, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    db_loader = DataLoader(db_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    model = SPCDH(args.bits, len(data.classes), **model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-05)
    info = {'identity': identity, 'model': model_params, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', 'split_counts': {k: len(v[0]) for k, v in data.splits.items()}, 'classes': data.classes, 'optimizer': 'AdamW', 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'minimum_lr': 1e-05, 'precision': 'float32', 'icmf_start_epoch': 21, 'icmf_weight': 0.3, 'triplets': 'batch other samples; intersecting labels positive; hardest positive/negative', 'adaptive_weight': 'all bits: zero through epoch 8; linear to 0.4 at 18; linear to 1 at 38', 'point_cloud': 'existing file unchanged; unit-normalized by the model during training', 'evaluation': {'query_split': 'test', 'database_split': 'val', 'candidates': 200, 'top_k': 100, 'selection': 'fixed final epoch; no test-based checkpoint selection', 'AP_denominator': 'relevant items within the returned top K (legacy repository)', 'ACG_WAP': 'shared-label counts; not percentages'}, 'smoke_test': args.smoke_test}
    atomic_json(output / 'run_info.json', info)
    checkpoint_path = output / 'last.pt'
    start_epoch, training_seconds = (1, 0.0)
    if checkpoint_path.exists() and (not args.smoke_test):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['identity'] != identity:
            raise ValueError('Checkpoint protocol mismatch; choose a new output root')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'])
        start_epoch = checkpoint['epoch'] + 1
        training_seconds = checkpoint['training_seconds']
        print(f'Resuming from epoch {start_epoch}', flush=True)
        del checkpoint
    if args.smoke_test:
        for epoch in (1, 21):
            losses = train_epoch(model, train_loader, optimizer, device, epoch, max_steps=1)
            print(json.dumps({'smoke_test': True, 'phase_epoch': epoch, 'losses': losses}), flush=True)
        finished_epoch = 0
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.monotonic()
            learning_rate = optimizer.param_groups[0]['lr']
            losses = train_epoch(model, train_loader, optimizer, device, epoch)
            scheduler.step()
            elapsed = time.monotonic() - started
            training_seconds += elapsed
            row = {'epoch': epoch, 'seconds': elapsed, 'lr': learning_rate, 'lambda_adaptive': adaptive_weight(epoch), **losses}
            print(json.dumps(row, allow_nan=False), flush=True)
            with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if epoch % 5 == 0 or epoch == args.epochs:
                save_checkpoint(checkpoint_path, {'identity': identity, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'training_seconds': training_seconds})
        finished_epoch = args.epochs
    started = time.monotonic()
    scores = evaluate(model, query_loader, db_loader, device)
    result = {'identity': identity, 'completed_epoch': finished_epoch, 'training_seconds': training_seconds, 'evaluation_seconds': time.monotonic() - started, 'queries': len(query_set), 'database': len(db_set), 'metrics': scores}
    atomic_json(result_path, result)
    print(json.dumps({'status': 'complete', **result}, ensure_ascii=False, allow_nan=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-params', type=Path, required=True)
    parser.add_argument('--bits', type=int, required=True, choices=(16, 32, 64, 128, 256))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/meitu')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or not 2 <= args.batch_size <= 256 or args.cpu_threads < 1:
        parser.error('epochs >= 1, 2 <= batch-size <= 256, cpu-threads >= 1 required')
    if args.smoke_test and args.output_root.resolve() == (ROOT / 'results').resolve():
        parser.error('--smoke-test requires a separate --output-root')
    output = args.output_root / f'{args.bits}bit'
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / 'running.lock'):
        run(args, output)
if __name__ == '__main__':
    main()
