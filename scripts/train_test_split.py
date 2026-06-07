import shutil
import os
import math

folders = os.listdir("dataset/cliped_videos")

for folder in folders:
    folder_path = f"dataset/cliped_videos/{folder}"
    folder_len = len(os.listdir(folder_path))
    split = math.ceil(folder_len * 0.8)
    i = 1
    for file in os.scandir(folder_path):
        file_path = f"dataset/train/videos/{file.name}"
        shutil.copy(f"{folder_path}/{file.name}", file_path)
        i = i + 1
        if i == split:
            break
    for file in os.scandir(folder_path):
        file_path = f"dataset/test/videos/{file.name}"
        shutil.copy(f"{folder_path}/{file.name}", file_path)
