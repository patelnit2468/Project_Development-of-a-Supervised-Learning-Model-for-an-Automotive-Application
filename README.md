# Project_Development-of-a-Supervised-Learning-Model-for-an-Automotive-Application
AIAS Project
# KITTI Car Detection using YOLO11

## Project Overview
This project focuses on detecting vehicles in urban environments using the **YOLO11n** architecture. The model was trained from scratch on the **KITTI Vision Benchmark Suite** to optimize performance for autonomous driving perception.

## Objectives
* **Targeted Detection:** Specialized focus on high-precision detection of the "Car" class.
* **Architecture Tuning:** Fine-tuning YOLO11 parameters (Learning Rate, Image Size, Batch Size) for the KITTI dataset.
* **Scratch Training:** Proof of model learning capacity without using pre-trained weights.

## Performance Metrics
The model was trained for **100 epochs** on an NVIDIA RTX 2050.
* **mAP50:** ~0.95 (95%)
* **True Positives:** 2,579
* **Training Stability:** Successfully converged with no overfitting.

## Key Results
### Accuracy and Loss
Detailed analysis of the training process shows a steady increase in $mAP50$ and a consistent drop in both Training and Validation loss.
<img width="1000" height="600" alt="Figure_100epoch" src="https://github.com/user-attachments/assets/476c0df8-0625-4ffe-903b-a813385aaca5" />



<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/9d60cb82-f663-4632-9cfd-ed7d548cccde" />

### Confusion Matrix
The model demonstrates high reliability in vehicle identification with minimal false negatives.

<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/47300973-f9af-46f4-a31a-927749a2eb4b" />


## Lessons Learned
1. **Path Management:** Implementation of absolute pathing to handle complex directory structures.
2. **Data Augmentation:** The impact of Mosaic Augmentation on initial learning and the benefit of disabling it for final fine-tuning (Epoch 90+).
3. **Resource Optimization:** Balancing batch sizes for 4GB VRAM hardware constraints.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run inference: `python scripts/predict.py --weights models/best.pt --source your_test_image.jpg`
