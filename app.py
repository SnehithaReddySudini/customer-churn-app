import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from churn_model import train_model
from sklearn.metrics import roc_curve, roc_auc_score
st.set_page_config(page_title="Customer Churn App")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below:")

# Load trained model
model, feature_names, accuracy, X_test, y_test = train_model()
col1, col2 = st.columns(2)

col1.metric("Model Accuracy", f"{accuracy:.2f}")
col2.metric("Model Type", "Random Forest")
# Sidebar
st.sidebar.header("Enter Customer Details")

input_data = {}

input_data["State"] = st.sidebar.selectbox(
    "State",
    ["OH", "NJ", "OK", "TX", "AL", "AK", "AZ", "CA"]
)

input_data["Area code"] = st.sidebar.selectbox(
    "Area Code",
    [408, 415, 510]
)


input_data["Number vmail messages"] = st.sidebar.number_input(
    "Number of Vmail Messages",
    value=0,
    key="vmail_messages"
)

input_data["Total day minutes"] = st.sidebar.number_input(
    "Total Day Minutes",
    value=200.0,
    key="total_day_minutes"
)
# ADD ALL OTHER INPUT FIELDS HERE
input_data["International plan"] = st.sidebar.selectbox(
    "International Plan",
    ["Yes", "No"]
)

input_data["Voice mail plan"] = st.sidebar.selectbox(
    "Voice Mail Plan",
    ["Yes", "No"]
)

input_data["Number vmail messages"] = st.sidebar.number_input("Number of Vmail Messages", value=0)

input_data["Total day minutes"] = st.sidebar.number_input("Total Day Minutes", value=200.0)
input_data["Total day calls"] = st.sidebar.number_input("Total Day Calls", value=100)
input_data["Total day charge"] = st.sidebar.number_input("Total Day Charge", value=30.0)

input_data["Total eve minutes"] = st.sidebar.number_input("Total Evening Minutes", value=200.0)
input_data["Total eve calls"] = st.sidebar.number_input("Total Evening Calls", value=100)
input_data["Total eve charge"] = st.sidebar.number_input("Total Evening Charge", value=20.0)

input_data["Total night minutes"] = st.sidebar.number_input("Total Night Minutes", value=200.0)
input_data["Total night calls"] = st.sidebar.number_input("Total Night Calls", value=100)
input_data["Total night charge"] = st.sidebar.number_input("Total Night Charge", value=10.0)

input_data["Total intl minutes"] = st.sidebar.number_input("Total International Minutes", value=10.0)
input_data["Total intl calls"] = st.sidebar.number_input("Total International Calls", value=3)
input_data["Total intl charge"] = st.sidebar.number_input("Total International Charge", value=3.0)

input_data["Customer service calls"] = st.sidebar.number_input("Customer Service Calls", value=1)
# Predict Button
if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    # Risk percentage
    risk_percent = probability * 100

    # Progress bar (Churn Risk Meter)
    st.write("Churn Risk Level")
    st.progress(int(risk_percent))

    # Display percentage
    st.write(f"Risk Score: {risk_percent:.2f}%")

    if prediction == 1:
        st.error("⚠ High Risk Customer")
    else:
        st.success("✅ Low Risk Customer")

    # Risk zone indicator
    if risk_percent < 30:
        st.success("Low Risk Zone 🟢")
    elif risk_percent < 70:
        st.warning("Medium Risk Zone 🟡")
    else:
        st.error("High Risk Zone 🔴")
st.subheader("Top 10 Feature Importance")

# Get feature importance from trained pipeline
importances = model.named_steps["classifier"].feature_importances_

# Get transformed feature names after OneHotEncoding
feature_names_transformed = model.named_steps["preprocessor"].get_feature_names_out()

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names_transformed,
    "Importance": importances
})

# Sort and take top 10
importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

# Plot
fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")

st.pyplot(fig)
st.subheader("ROC Curve")

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
ax2.plot([0, 1], [0, 1])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"ROC Curve (AUC = {auc_score:.2f})")

st.pyplot(fig2)