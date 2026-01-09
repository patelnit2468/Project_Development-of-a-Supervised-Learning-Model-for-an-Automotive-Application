import os
import random
import shutil
from tqdm import tqdm

# Paths
BASE_DIR = "data"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels_yolo")

# Output Structure
OUTPUT_DIR = "kitti_split"
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# Get all file IDs (filenames without extension)
all_ids = [os.path.splitext(f)[0] for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
random.shuffle(all_ids) # Random sampling for representativeness

# Calculate split points
train_end = int(0.8 * len(all_ids))
val_end = int(0.9 * len(all_ids))

train_ids = all_ids[:train_end]
val_ids = all_ids[train_end:val_end]
test_ids = all_ids[val_end:]

def move_files(ids, split):
    for name in tqdm(ids, desc=f"Moving {split}"):
        # Move Image
        shutil.copy(os.path.join(IMAGES_DIR, name + ".png"), 
                    os.path.join(OUTPUT_DIR, split, 'images', name + ".png"))
        # Move Label
        shutil.copy(os.path.join(LABELS_DIR, name + ".txt"), 
                    os.path.join(OUTPUT_DIR, split, 'labels', name + ".txt"))

move_files(train_ids, 'train')
move_files(val_ids, 'val')
move_files(test_ids, 'test')