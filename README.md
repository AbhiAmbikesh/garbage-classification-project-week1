# Garbage Classification AI Model (Week 1)

This project is built as part of the AICTE Internship (Week 1) under Edunet Foundation.

## 🔍 Objective
Build a Convolutional Neural Network (CNN) model to classify garbage images into 6 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## 📂 Dataset
- Source: Kaggle
- Structure: Images are organized in `train/` and `validation/` folders
- Each folder contains subfolders for every class

## 🧠 Model Architecture
- Framework: TensorFlow + Keras
- Layers: 3 Convolutional + MaxPooling + Flatten + Dense
- Final Activation: Softmax
- Accuracy Achieved: ~55%

## 🧪 How to Test
1. Save any image as `test_image.jpg`
2. Run the test script:
```bash
python test_model.py
