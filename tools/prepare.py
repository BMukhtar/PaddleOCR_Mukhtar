import os
import json
import random
from tqdm import tqdm

# set random seed
random.seed(0)
test_size = 5000

def convert_to_train_test_format(ann_folder, img_folder, train_file, test_file, train_test_split_ratio=0.8):
    # Initialize an empty list to hold tuples of image filenames and their corresponding labels
    data = []
    
    # Iterate over all json files in the annotation folder
    for ann_file in tqdm(os.listdir(ann_folder), desc="Reading annotation files"):
        # Ensure that the file is indeed a json file
        if ann_file.endswith('.json'):
            # Open the json file and load its contents
            with open(os.path.join(ann_folder, ann_file), 'r') as f:
                ann_data = json.load(f)
                
                # Append a tuple of the image filename and its label to the list
                data.append((ann_data['name'].strip(), ann_data['description'].strip()))
    
    # Shuffle the data to ensure a good mix of different examples in the training and test sets
    random.shuffle(data)
    
    # Split the data into training and test sets
    train_data = data[test_size:]
    test_data = data[:test_size]
    
    # Write the training data to the train_file
    with open(train_file, 'w') as f:
        for img_file, label in tqdm(train_data, desc="Writing training file"):
            f.write(f"{img_file}\t{label}\n")
            
    # Write the test data to the test_file
    with open(test_file, 'w') as f:
        for img_file, label in tqdm(test_data, desc="Writing testing file"):
            f.write(f"{img_file}\t{label}\n")

base_path = "./train_data/HK_dataset/"
# Path to the annotation and image folders
ann_folder = base_path + 'ann'
img_folder = base_path + 'img'

# Path to the output train and test files
train_file = base_path + 'hk_train.txt'
test_file = base_path + 'hk_test.txt'

if __name__ == "__main__":
    convert_to_train_test_format(ann_folder, img_folder, train_file, test_file)


