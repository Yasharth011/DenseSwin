import os
from re import split
import subprocess


def get_video_duration(video_path):
    """Queries FFmpeg to get the exact duration of a video file in seconds"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    return float(result.stdout.strip())

def slice_video_physical(video_dir, output_root, chunk_duration):
    video_files = [f for f in os.listdir(video_dir)]

    for video_file in video_files:

        video_name, video_type = os.path.splitext(video_file)
        video_path = os.path.join(video_dir, video_file)
        target_dir = os.path.join(output_root, video_name)
        os.makedirs(target_dir, exist_ok=True)

        duration = get_video_duration(video_path)
        start_time = 0.0
        clip_counter = 0

        print(f"Physically slicing {video_file} ({duration} seconds)...")

        while start_time + chunk_duration <= duration:
            output_clip_name = f"{video_name}_{clip_counter}{video_type}"
            output_clip_path = os.path.join(target_dir, output_clip_name)

            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-t",
                str(chunk_duration),
                "-c",
                "copy",
                "-loglevel",
                "error",
                output_clip_path,
            ]
            subprocess.run(cmd)

            start_time += chunk_duration
            clip_counter += 1

    print(f"Successfully generated physical clips inside {output_root}/")


slice_video_physical(
    video_dir="dataset/videos",
    output_root="dataset/cliped_videos",
    chunk_duration=2.0,
)
