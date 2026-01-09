import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import os
import glob

# 1. Setup Paths - UPDATE THESE TO YOUR ACTUAL PATHS
model_path = r"D:\THI\AIAS_Project\dataset\data_object_image_2\training\kitti_split\runs\detect\kitti_car_final\weights\best.pt"
test_images_path = r"D:\THI\AIAS_Project\dataset\data_object_image_2\training\runs\detect\predict\*.jpg" # or .png

# 2. Load Model
model = YOLO(model_path)

# 3. Get first 4 images from the test folder
image_list = glob.glob(test_images_path)[:4]

# 4. Create Plotting Grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, img_path in enumerate(image_list):
    # Run prediction
    results = model.predict(source=img_path, conf=0.5)[0]
    
    # Get annotated image (BGR) and convert to RGB
    annotated_img = results.plot()
    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Display in subplot
    axes[i].imshow(rgb_img)
    axes[i].set_title(f"Test Image: {os.path.basename(img_path)}")
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle("YOLO Detection Results on KITTI Test Set", fontsize=20, y=1.02)
plt.savefig("test_predictions_grid.png", dpi=300)
plt.show()