from dotenv import dotenv_values
from dataclasses import dataclass
import os 

config = dotenv_values('.env')

@dataclass
class DatasetConfig: 
    parent_path= str(config['DATASET'])
    videos = os.path.join(parent_path, 'videos')
    frames = os.path.join(parent_path, 'frames')
    frames_csv = os.path.join(parent_path, 'points.csv')

DATASET_CONFIG = DatasetConfig()
