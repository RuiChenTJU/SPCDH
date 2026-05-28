

import os
import cv2
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import time


def extract_frames(video_path, output_dir, num_frames=16):
    try:

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            return False

        if total_frames <= num_frames:

            frame_indices = list(range(total_frames))
        else:

            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]

        saved_count = 0
        for idx, frame_idx in enumerate(frame_indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:

                output_path = os.path.join(output_dir, f"{idx:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1

        cap.release()

        if saved_count < num_frames:
            return False

        return True

    except Exception as e:
        print("SPCDH")
        print("SPCDH")
        return False


def process_single_video(args):
    video_path, output_dir, num_frames = args

    if os.path.exists(output_dir):
        frame_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        if len(frame_files) >= num_frames:
            return (True, video_path)

    success = extract_frames(video_path, output_dir, num_frames)
    return (success, video_path)


def process_videos(video_root, image_root, num_frames=16, num_workers=None):
    subdirs = ['train', 'val', 'test']

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    print("SPCDH")

    total_videos = 0
    success_count = 0
    failed_count = 0

    for subdir in subdirs:
        video_dir = os.path.join(video_root, subdir)
        image_dir = os.path.join(image_root, subdir)

        if not os.path.exists(video_dir):
            print("SPCDH")
            continue

        os.makedirs(image_dir, exist_ok=True)

        video_files = []
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_files.append(filename)

        total_videos += len(video_files)

        print("SPCDH")

        tasks = []
        for filename in video_files:
            video_path = os.path.join(video_dir, filename)
            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(image_dir, video_name)
            tasks.append((video_path, output_dir, num_frames))

        start_time = time.time()
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_video, tasks),
                total=len(tasks),
                desc="SPCDH"
            ))

        for success, video_path in results:
            if success:
                success_count += 1
            else:
                failed_count += 1

        elapsed = time.time() - start_time
        print("SPCDH")

    print("\n" + "=" * 60)
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("=" * 60)


if __name__ == "__main__":

    VIDEO_ROOT = "/media/Backup/cr/dataset_hash/video"
    IMAGE_ROOT = "/media/Backup/cr/dataset_hash/image"
    NUM_FRAMES = 16

    print("=" * 60)
    print("SPCDH")
    print("=" * 60)
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("=" * 60)

    if not os.path.exists(VIDEO_ROOT):
        print("SPCDH")
        exit(1)

    process_videos(VIDEO_ROOT, IMAGE_ROOT, NUM_FRAMES, num_workers=None)

    print("\n" + "=" * 60)
    print("SPCDH")
    print("=" * 60)
