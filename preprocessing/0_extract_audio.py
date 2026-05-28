import os
from moviepy.editor import VideoFileClip


def extract_audio_from_videos(video_root, audio_root):

    if not os.path.exists(audio_root):
        os.makedirs(audio_root)
        print(f"Created audio root directory: {audio_root}")

    subdirs = ['train', 'test', 'val']

    for subdir in subdirs:
        video_dir = os.path.join(video_root, subdir)
        audio_dir = os.path.join(audio_root, subdir)

        if not os.path.exists(video_dir):
            print(f"Warning: Video directory not found: {video_dir}")
            continue

        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            print(f"Created audio directory: {audio_dir}")

        print(f"Processing directory: {subdir}...")
        files = os.listdir(video_dir)

        for filename in files:

            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_path = os.path.join(video_dir, filename)

                name_without_ext = os.path.splitext(filename)[0]
                audio_filename = name_without_ext + ".mp3"
                audio_path = os.path.join(audio_dir, audio_filename)

                if os.path.exists(audio_path):
                    print(f"  Skipping (already exists): {audio_filename}")
                    continue

                try:

                    video = VideoFileClip(video_path)
                    if video.audio is not None:

                        video.audio.write_audiofile(audio_path, logger=None, verbose=False)
                        print(f"  Extracted: {filename} -> {audio_filename}")
                    else:
                        print(f"  No audio track found in: {filename}")

                    video.close()

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")


if __name__ == "__main__":

    VIDEO_ROOT_DIR = "/media/Backup/cr/dataset_hash/video"
    AUDIO_ROOT_DIR = "/media/Backup/cr/dataset_hash/audio"

    print("Starting audio extraction...")
    extract_audio_from_videos(VIDEO_ROOT_DIR, AUDIO_ROOT_DIR)
    print("All done!")
