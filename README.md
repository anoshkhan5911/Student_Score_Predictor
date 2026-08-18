# 🎓 AI Student Performance Predictor


## 📌 Key Highlights

- **Predictive ML Pipeline:** Utilizes a trained **Gradient Boosting Regressor** with scaled input vectors to forecast overall percentages without relying on past grade history.
- **Data Preprocessing & Scaling:** Includes robust categorical encoding and `StandardScaler` transformations saved as standalone pipeline assets.
- **Interactive Web Interface:** Features a clean, responsive UI with real-time feedback, progress indicators, and dynamic advice based on predicted score thresholds.
- **Production Ready:** Fully configured with structured dependencies for one-click deployment on Streamlit Cloud.

---

## 🛠️ Tech Stack

| Domain | Technologies / Libraries |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Machine Learning** | Scikit-Learn, NumPy, Pandas |
| **Model Persistence** | Joblib |
| **Data Visualization & EDA** | Matplotlib, Seaborn |
| **Web Framework & Deployment** | Streamlit, Streamlit Community Cloud |

---

## 📂 Project Structure

```text
Student_Score_Predictor/
│
├── Student_Analysis.ipynb     # Exploratory Data Analysis & Model Training
├── Student_Performance.csv    # Dataset used for training and testing
├── student_model.pkl          # Serialized Gradient Boosting Model
├── scaler.pkl                 # Fitted StandardScaler object
├── model_features.pkl         # Feature names mapping vector
├── app.py                     # Streamlit frontend & inference logic
├── requirements.txt           # Python dependencies for deployment
└── README.md                  # Project documentation