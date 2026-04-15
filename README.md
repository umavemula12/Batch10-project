🧠 Intracranial Hemorrhage (ICH) Detection System

This project is a deep learning-based web application designed to detect and classify Intracranial Hemorrhage (ICH) from brain CT scan images. It uses a hybrid model combining classification and segmentation techniques to provide accurate results.

📌 Project Overview
Detects brain hemorrhage from CT scan images
Classifies different hemorrhage types
Uses deep learning models (CNN + U-Net)
Provides a user-friendly web interface with Streamlit integration

The system helps in faster and more accurate medical diagnosis.

📂 Dataset

The project uses brain CT scan datasets such as:

RSNA Intracranial Hemorrhage Dataset
Brain CT Scan Dataset

These datasets contain multiple CT slices with labels indicating hemorrhage presence and types.

👉 Step 1:
Download the dataset and place it in your Google Drive.

⚙️ Model Training

👉 Step 2:
Open and run the notebook:

train_models.ipynb
Run all cells one by one in Google Colab
This trains both classification and segmentation models
💾 Download Trained Models

👉 Step 3:
After training, download the following files:

best_cls_model.keras (Classification model)
ich_segmentation_unet.keras (Segmentation model)

You can download them from:

Google Drive OR
Colab local files
📁 Project Setup

👉 Step 4:

Open the web project in VS Code:

Batch-10_BSec_Project_Documents/
└── Project Source Code/
    └── WEB/
        └── website_pages/
            └── backend/

👉 Paste the downloaded model files here:

backend/
├── best_cls_model.keras
├── ich_segmentation_unet.keras
🚀 Run the Application

👉 Step 5: Open terminal and run:

conda create -n ich_app python=3.10 -y
conda activate ich_app

pip install --upgrade pip
pip install streamlit tensorflow opencv-python pydicom matplotlib numpy

streamlit run test.py
🌐 Web Interface

👉 Step 6:

Open the main web page:

Batch-10_BSec_Project_Documents/
└── Project Source Code/
    └── WEB/
        └── website_pages/
            └── index.html

This will display the project website.

🧪 How to Use
Click on Model Deployment
Streamlit app will open inside the page
Upload a CT scan image
The model will:
Detect hemorrhage
Display prediction results

The deployment is integrated using a local Streamlit server.

🧠 Model Architecture
CNN-based feature extraction
ResNet-50 for classification
U-Net for segmentation
Hybrid approach improves accuracy and localization

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/5ee890d5-f3d8-4146-ab84-5b424b7ac0ea" />


🔄 Preprocessing Techniques
Multi-window CT image enhancement
Adjacent slice-based encoding
Region-of-interest extraction

These steps improve model performance and accuracy.

📊 Results
Accuracy: 95%
AUC: 0.96
High precision and reliable predictions

<img width="881" height="515" alt="image" src="https://github.com/user-attachments/assets/bf34a04c-9db1-490c-b056-5e5483088617" />
<img width="889" height="594" alt="image" src="https://github.com/user-attachments/assets/53821ac3-45e5-4b2c-97a4-bc6c1e707fd5" />
<img width="883" height="640" alt="image" src="https://github.com/user-attachments/assets/66906ed0-8610-493a-bf63-993400f4a60e" />
<img width="783" height="419" alt="image" src="https://github.com/user-attachments/assets/796131c2-fa2b-4e51-b94a-6aebe5150c99" />
<img width="906" height="676" alt="image" src="https://github.com/user-attachments/assets/97673dc9-900f-48c0-8513-04d62528062b" />
<img width="644" height="418" alt="image" src="https://github.com/user-attachments/assets/1fd3286f-0842-4284-bf31-639138cf06b2" />


🔮 Future Scope
Improve model accuracy with more data
Real-time hospital integration
Advanced AI-based diagnostics

👨‍💻 Team
Vemula Uma
Seelam Bhanu
Bhukya Abhiram
Guide: Dr. Narisetty Srinivasa Rao
✅ Conclusion

This project demonstrates how AI and deep learning can assist in early detection of brain hemorrhage, helping doctors make faster and more accurate decisions.
