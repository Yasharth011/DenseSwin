from decord import VideoReader
from decord import cpu
from argparse import ArgumentParser
from PIL import Image
import os

parser = ArgumentParser(
        prog="extract_frames",
        description="extract frames from videos"
        )
parser.add_argument('-v', '--videos', help='path of video-frames')
parser.add_argument('-f', '--frames', help='path to store frames')

args = parser.parse_args()

try: 
    for file in os.scandir(args.videos):
        print(f'Extracting video {file.name}...')
        if file.is_file():
            vr = VideoReader(file.path, ctx=cpu(0))
            for i in range(len(vr)):
                file_name = f'{(file.name).split('.')[0]}_{i}.png'
                img = Image.fromarray(vr[i].asnumpy())
                img.save(args.frames+file_name)

except Exception as e: 
    print(f'Error: {str(e)}')

            
