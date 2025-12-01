🧠 Brain Tumor Classification Using Convolutional Neural Networks (CNN)
This project uses deep learning to detect and classify brain tumors from MRI images. The model is built using TensorFlow/Keras, following a complete ML pipeline that includes data preprocessing, CNN model development, training, evaluation, and prediction.


🚀 Project Overview
Brain tumors are a critical medical condition requiring early and accurate diagnosis. This project leverages a Convolutional Neural Network (CNN) to classify MRI images into categories such as:
Glioma
Meningioma
Pituitary
No Tumor


The goal is to build a model that can assist radiologists by providing fast and reliable predictions.
🧩 Features
✔️ MRI image preprocessing
✔️ Custom CNN architecture
✔️ Training with data augmentation
✔️ Evaluation using accuracy, confusion matrix, and loss curves
✔️ Predictive inference on new MRI images
✔️ Well-documented notebook with step-by-step workflow

🧠 Model Architecture (CNN)
The model consists of:
Convolutional layers with ReLU activation
MaxPooling layers
Dropout for regularization
Flatten + Dense fully connected layers
Softmax classification output
This architecture is optimized for image classification tasks.


📥 Dataset
The dataset used contains labeled MRI images.
If you're using the popular Kaggle dataset, it can be found here:
Brain Tumor MRI Dataset
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
Due to size restrictions, the dataset is not included in the repository.

🔮 How the Model Predicts Tumors
Image is resized to target shape (e.g., 150×150 or 224×224)
Normalized to pixel values between 0–1
Passed through CNN layers to extract spatial features
Dense layers classify tumor type
Output = predicted class with probability score


📌 Future Improvements
Add Grad-CAM for visual explanation
Deploy the model using Flask or FastAPI
Build a React or mobile interface
Improve accuracy with Transfer Learning (ResNet, VGG16, etc.)
