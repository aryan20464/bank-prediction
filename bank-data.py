import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App
st.title("Bank Deposit Prediction App")

# Step 1: Upload Training CSV
st.sidebar.header("Upload Data")
uploaded_train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])

if uploaded_train_file:
    train_data = pd.read_csv(uploaded_train_file)
    st.write("Training Data Preview")
    st.dataframe(train_data.head())
    
    # Data Preprocessing
    st.subheader("Data Preprocessing")
    train_data['y'] = train_data['y'].apply(lambda x: 1 if x == "yes" else 0)
    categorical_cols = train_data.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols.remove('y')
    train_data = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
    X = train_data.drop('y', axis=1)
    y = train_data['y']
    
    # Model Training
    st.subheader("Model Training")
    model_type = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    r2 = r2_score(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write("Classification Report (Validation Data)")
    st.text(classification_report(y_val, y_pred_val))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_val, y_pred_val)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    st.pyplot(plt)

    # Step 2: Upload Test Data
    st.sidebar.header("Upload Test Data")
    uploaded_test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])
    
    if uploaded_test_file:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("Test Data Preview")
        st.dataframe(test_data.head())
        
        # Apply preprocessing to test data
        test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)
        missing_cols = set(X.columns) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        test_data = test_data[X.columns]
        
        # Make predictions
        st.subheader("Prediction Results")
        test_predictions = model.predict(test_data)
        test_data['Prediction'] = test_predictions
        st.write("Prediction Results")
        st.dataframe(test_data[['Prediction']])
        
        # Visualization
        st.subheader("Prediction Visualization")
        sns.countplot(x=test_data['Prediction'], palette="viridis")
        st.pyplot(plt)
