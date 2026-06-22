import pandas as pd
import json
from utils import TEST_DATASET, TRAIN_DATASET

def find_global_max_cars(csv_path):
    df = pd.read_csv(csv_path)
    
    max_cars_overall = 0
    video_with_max = ""
    
    for index, row in df.iterrows():
        try:
            bboxes = json.loads(row['bounding_boxes'])
            
            for frame_boxes in bboxes:
                num_cars = len(frame_boxes) // 4
                
                if num_cars > max_cars_overall:
                    max_cars_overall = num_cars
                    video_with_max = row['file_name']
                    
        except Exception as e:
            print(f"Error parsing row {index} ({row.get('file_name', 'Unknown')}): {e}")
            
    print(f"Maximum Cars in a Single Frame: {max_cars_overall}")
    print(f"Found in video: {video_with_max}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="create train or test dataset")
    args = parser.parse_args()

    if args.dataset == "train":
        find_global_max_cars(TRAIN_DATASET.csv)
    elif args.dataset == "test":
        find_global_max_cars(TEST_DATASET.csv)
