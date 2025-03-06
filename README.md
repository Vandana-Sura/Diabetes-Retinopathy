## 📌 Overview
This project is a **Deep Learning-based Diabetic Retinopathy Detection System** that classifies the severity of **diabetic retinopathy** from **retinal images**. The model is trained on a dataset of fundus images and predicts the stage of the disease to assist in early diagnosis and treatment planning.

## 🚀 Features
- **Deep Learning Model**: Uses CNN-based architecture for image classification.
- **Automated Diagnosis**: Predicts different severity levels of diabetic retinopathy.
- **User-Friendly Interface**: Hosted on **Hugging Face Spaces** for easy accessibility.
- **Real-Time Analysis**: Upload an image and get instant predictions.
- **Camera Access**: Capture retinal images directly from your device's camera.

## 🏗 Tech Stack
- **Framework**: TensorFlow/Keras
- **Frontend**: Gradio (for UI)
- **Dataset**: Kaggle’s Diabetic Retinopathy dataset
- **Deployment**: Hugging Face Spaces
- **Camera Integration**: HTML5/WebRTC for real-time image capture

## 🔧 Installation
### Clone the repository:
```bash
git clone https://github.com/Vandana-Sura/Diabetes-Retinopathy.git
cd Diabetes-Retinopathy
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the application locally:
```bash
python app.py
```

## 📌 Usage
1. Upload a **retinal image** or capture one using your **camera**.
2. The model will analyze and classify the **severity of diabetic retinopathy**.
3. Get **instant results** with confidence scores.
