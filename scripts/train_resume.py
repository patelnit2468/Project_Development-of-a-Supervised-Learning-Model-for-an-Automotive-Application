from ultralytics import YOLO

# 1. Load your specific saved progress
model = YOLO(r"D:\THI\AIAS_Project\dataset\data_object_image_2\training\kitti_split\runs\detect\kitti_car_final\weights\last.pt")

# 2. Resume training
# This will automatically pick up from Epoch 15 and continue to 100
model.train(resume=True)