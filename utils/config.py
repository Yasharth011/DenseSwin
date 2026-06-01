from dotenv import dotenv_values
from dataclasses import dataclass
import os 

config = dotenv_values('.env')

@dataclass
class DatasetConfig: 
    parent_path= str(config['DATASET'])
    videos = os.path.join(parent_path, 'videos')
    frames = os.path.join(parent_path, 'frames')
    annotated_frames = os.path.join(parent_path, 'annotated_frames')
    frames_csv = os.path.join(parent_path, 'bboxes.csv')

DATASET_CONFIG = DatasetConfig()
