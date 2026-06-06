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
    csv_dir = os.path.join(parent_path, 'csv')
    main_csv = os.path.join(parent_path, 'main.csv')

@dataclass
class ModelConfig:
    parent_path= str(config['MODEL'])
    logs = os.path.join(parent_path, 'logs')
    checkpoints = os.path.join(parent_path, 'checkpoints')

DATASET_CONFIG = DatasetConfig()
MODEL_CONFIG = ModelConfig()
