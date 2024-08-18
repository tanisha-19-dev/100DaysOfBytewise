import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Data Cleaning
def clean_data(df):
    df_cleaned = df.drop(columns=['Subcategory', 'Note.1', 'Account.1', 'Amount', 'Currency'])
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    return df_cleaned

# Feature Engineering
def feature_engineering(df_cleaned):
    df_cleaned['Month'] = df_cleaned['Date'].dt.month
    df_cleaned['Year'] = df_cleaned['Date'].dt.year
    df_encoded = pd.get_dummies(df_cleaned, columns=['Account', 'Income/Expense'], drop_first=True)
    return df_encoded

# Model Training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

# Main Streamlit App
def main():
    st.title("Expense Tracker with Automatic Categorization")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("### Raw Data")
        st.write(df.head())

        df_cleaned = clean_data(df)
        df_encoded = feature_engineering(df_cleaned)

        X = df_encoded.drop(columns=['Date', 'Category', 'Note'])
        y = df_encoded['Category']

        model, accuracy, report = train_model(X, y)

        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(report)

        # Real-time Expense Prediction
        st.write("### Real-time Expense Prediction")
        account = st.selectbox("Select Account", options=df_cleaned['Account'].unique())
        inr = st.number_input("Enter Amount (INR)", min_value=0.0, step=1.0)
        month = st.selectbox("Select Month", options=range(1, 13))
        year = st.selectbox("Select Year", options=df_cleaned['Year'].unique())
        income_expense = st.selectbox("Income or Expense", options=['Expense', 'Income'])

        if st.button("Predict Category"):
            account_encoded = [1 if acc == account else 0 for acc in df_encoded.columns if acc.startswith("Account_")]
            income_expense_encoded = 1 if income_expense == 'Income' else 0
            input_data = [inr, month, year] + account_encoded + [income_expense_encoded]
            prediction = model.predict([input_data])
            st.write(f"The predicted category is: **{prediction[0]}**")

if __name__ == "__main__":
    main()
