📊 Customer Churn Prediction Web App

An end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn.

This project demonstrates data preprocessing, model training, evaluation, and deployment using an interactive web interface.

🚀 Live Demo

🔗 Deployed App:
https://customer-churn-app-swqfszfjm3oj7azankkffr.streamlit.app/
Built and deployed using Streamlit Cloud.

📌 Problem Statement

Customer churn is a major challenge in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds a predictive model to:

Identify customers at high risk of churn

Provide churn probability scores

Help businesses take proactive retention measures

🧠 Machine Learning Pipeline

The project follows a complete ML workflow:

1️⃣ Data Preprocessing

Handled categorical features using encoding

Used ColumnTransformer for structured preprocessing

Train-test split for unbiased evaluation

2️⃣ Model Training

Algorithm: Random Forest Classifier

Implemented using Scikit-learn

Trained on telecom churn dataset

3️⃣ Model Evaluation

Accuracy score

ROC Curve visualization

Feature importance analysis

📊 Model Performance

✅ Accuracy: ~0.90+ (depending on dataset split)

📈 ROC Curve plotted

🔍 Feature importance displayed in dashboard

💻 Application Features

Interactive customer input form

Real-time churn prediction

Churn probability score

Risk level indicator (Low / Medium / High)

Feature importance visualization

ROC curve display

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Streamlit

📂 Project Structure
customer-churn-app/
│
├── app.py                  # Streamlit application
├── churn_model.py          # ML training pipeline
├── churn-bigml-20.csv      # Dataset
├── requirements.txt        # Dependencies
└── README.md
▶️ How To Run Locally

Clone the repository:

git clone https://github.com/SnehithaReddySudini/customer-churn-app.git
cd customer-churn-app

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
🌍 Deployment

The application is deployed on Streamlit Cloud and publicly accessible via the live demo link above.

📈 Business Impact

This system can help telecom companies:

Reduce customer churn

Improve customer retention strategies

Identify high-risk customer segments

Optimize marketing campaigns

🚀 Future Improvements

Compare multiple models (Logistic Regression, XGBoost)

Add SHAP for model explainability

Implement model persistence using Pickle

Add database integration

Containerize using Docker

👨‍💻 Author

SnehithaReddySudini
Aspiring Data Scientist
GitHub: https://github.com/SnehithaReddySudini
