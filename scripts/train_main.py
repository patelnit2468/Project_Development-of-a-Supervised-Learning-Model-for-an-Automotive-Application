from ultralytics import YOLO
import torch

def main():
    # 1. PATH TO YOUR LAST CHECKPOINT
    # This file contains the model's "memory" of where it stopped
    checkpoint_path = r"D:\THI\AIAS_Project\dataset\data_object_image_2\training\kitti_split\runs\detect\kitti_car_final\weights\last.pt"
    
    # 2. LOAD THE MODEL
    model = YOLO(checkpoint_path)

    # 3. RESUME WITH A NEW LIMIT
    results = model.train(
        resume=True,
        epochs=model.ckpt['epoch'] + 11, # Current epoch + 10 extra (plus 1 for index)
        workers=0                        # Keeps it stable on Windows
    )

if __name__ == '__main__':
    main()