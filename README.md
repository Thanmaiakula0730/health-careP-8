Perfect ✅
Here’s your **updated README.md** with an **example code snippet** showing how to load `medical_cnn_model.h5` and make predictions using new patient data.

---

# 🏥 HealthCare Project – CNN Model for Medical Test Prediction

## 📌 Overview

This project builds a **Convolutional Neural Network (CNN)** model to predict **medical test results** based on patient information such as **Name, Age, Gender, Blood Type, Medical Condition, and Medication**.

The dataset is preprocessed, encoded, and standardized before being fed into the model. The trained model is saved as `medical_cnn_model.h5` for future predictions.

---

## 📂 Project Structure

```
📁 HealthCare-Project-8
│── healthcare_dataset.csv        # Dataset containing patient information
│── train.py                      # Python script to train the CNN model
│── medical_cnn_model.h5          # Saved trained CNN model
│── README.md                     # Project documentation
```

---

## 🛠️ Requirements

Make sure you have Python installed along with the following dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow
```

---

## 📊 Dataset

The dataset (`healthcare_dataset.csv`) contains columns:

* **Name** – Patient’s name
* **Age** – Patient’s age
* **Gender** – Male/Female
* **Blood Type** – e.g., A+, O-, etc.
* **Medical Condition** – Disease/condition of the patient
* **Medication** – Prescribed medication
* **Test Results** – Target label (e.g., Positive, Negative)

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Thanmaiakula0730/health-careP-8.git
cd health-careP-8
```

### 2️⃣ Run the training script

```bash
python train.py
```

**Output:**

* The model will be trained and saved as `medical_cnn_model.h5`.
* Sample predictions will be displayed in the terminal.

---

## 🧠 Model Architecture

The CNN model includes:

* **Conv1D layer** (32 filters, kernel size 2, ReLU activation)
* **MaxPooling1D layer**
* **Flatten layer**
* **Dense layer** (64 units, ReLU activation)
* **Dropout layer** (0.3)
* **Dense output layer** (softmax activation)

---

## 📈 Training

* **Optimizer:** Adam
* **Loss:** Categorical Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 50
* **Batch Size:** 4
* **Validation Split:** 10%

---

## 🏆 Sample Output

```
Sample predictions:
Predicted: Positive | True: Positive
Predicted: Negative | True: Negative
...
```

---

## 🔮 Example: Load Model & Make Predictions

Here’s how you can use the trained model for new patient data:

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  # Optional: if you save encoders

# Load model
model = load_model("medical_cnn_model.h5")

# Example input data
new_patient = pd.DataFrame([{
    "Name": "John Doe",
    "Age": 45,
    "Gender": "Male",
    "Blood Type": "O+",
    "Medical Condition": "Diabetes",
    "Medication": "Metformin"
}])

# Load dataset to fit encoders
dataset = pd.read_csv("healthcare_dataset.csv")

# Encode categorical columns
label_encoders = {}
for col in ["Name", "Gender", "Blood Type", "Medical Condition", "Medication"]:
    le = LabelEncoder()
    le.fit(dataset[col])
    new_patient[col] = le.transform(new_patient[col])
    label_encoders[col] = le

# Scale Age
scaler = StandardScaler()
scaler.fit(dataset[["Age"]])
new_patient["Age"] = scaler.transform(new_patient[["Age"]])

# Reshape for CNN input
X_input = new_patient.to_numpy().astype(np.float32).reshape((1, new_patient.shape[1], 1))

# Make prediction
prediction = model.predict(X_input)
predicted_class = np.argmax(prediction, axis=1)

# Decode target labels
target_encoder = LabelEncoder()
target_encoder.fit(dataset["Test Results"])
print("Predicted Test Result:", target_encoder.inverse_transform(predicted_class)[0])
```

---

## 📌 Future Improvements

* Add a **Streamlit or Flask web interface** for easy predictions.
* Increase dataset size for better accuracy.
* Experiment with **hyperparameter tuning**.

---

## 👩‍💻 Author

**Akula Thanmai**
📧 Contact: *\[Your Email]*
🔗 GitHub: [Thanmaiakula0730](https://github.com/Thanmaiakula0730)

---

If you want, I can also **include the code to automatically save your label encoders and scaler** during training so that the prediction code won’t need to refit them every time. That will make the deployment cleaner and faster.
