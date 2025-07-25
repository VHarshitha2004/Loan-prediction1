import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("loan prediction.csv")

    # Clean currency fields
    df['customer_income'] = df['customer_income'].str.replace('£', '').str.replace(',', '').astype(float)
    df['loan_amnt'] = df['loan_amnt'].str.replace('£', '').str.replace(',', '').astype(float)

    # Handle missing values
    df = df.dropna(subset=['Current_loan_status'])
    df['employment_duration'] = df['employment_duration'].fillna(df['employment_duration'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df['historical_default'] = df['historical_default'].fillna('N')
    df['Current_loan_status'] = df['Current_loan_status'].map({'DEFAULT': 1, 'NO DEFAULT': 0})

    # Drop ID
    df = df.drop(columns=['customer_id'])

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)
    return df

# Load data
df = load_data()
X = df.drop("Current_loan_status", axis=1)
y = df["Current_loan_status"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --------- STREAMLIT UI ---------
st.title("🏦 Loan Prediction Web App")
st.markdown("Fill in the customer details below to predict whether the loan will be **approved** or **rejected**.")

# Input form
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (£)", min_value=0.0, value=50000.0)
home = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
emp_duration = st.number_input("Employment Duration (months)", min_value=0, value=12)
intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'PERSONAL'])
grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E'])
loan_amnt = st.number_input("Loan Amount (£)", min_value=0.0, value=10000.0)
loan_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0)
term = st.selectbox("Term (Years)", [1, 5, 10])
past_default = st.selectbox("Any Past Default?", ['Y', 'N'])
cred_length = st.number_input("Credit History Length (years)", min_value=0, value=3)

# Prepare input
input_dict = {
    'customer_age': age,
    'customer_income': income,
    'employment_duration': emp_duration,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_rate,
    'term_years': term,
    'cred_hist_length': cred_length,
    'home_ownership_OTHER': 1 if home == 'OTHER' else 0,
    'home_ownership_OWN': 1 if home == 'OWN' else 0,
    'home_ownership_RENT': 1 if home == 'RENT' else 0,
    'loan_intent_EDUCATION': 1 if intent == 'EDUCATION' else 0,
    'loan_intent_HOMEIMPROVEMENT': 1 if intent == 'HOMEIMPROVEMENT' else 0,
    'loan_intent_MEDICAL': 1 if intent == 'MEDICAL' else 0,
    'loan_intent_PERSONAL': 1 if intent == 'PERSONAL' else 0,
    'loan_intent_VENTURE': 1 if intent == 'VENTURE' else 0,
    'loan_grade_B': 1 if grade == 'B' else 0,
    'loan_grade_C': 1 if grade == 'C' else 0,
    'loan_grade_D': 1 if grade == 'D' else 0,
    'loan_grade_E': 1 if grade == 'E' else 0,
    'historical_default_Y': 1 if past_default == 'Y' else 0
}

# Add missing columns (that might not appear based on user input)
for col in X.columns:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(input_df)[0]
    
    if prediction == 0:
        st.success("✅ Loan Prediction: Approved")
        st.markdown("The applicant is likely to repay the loan.")
    else:
        st.error("❌ Loan Prediction: Rejected")
        st.markdown("The applicant is likely to default on the loan.")
