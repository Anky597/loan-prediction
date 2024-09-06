import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

def main():
    st.title("Loan Approval Prediction")

    with st.form(key='loan_form'):
        no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
        income_annum = st.number_input('Annual Income', min_value=0, value=0)
        loan_amount = st.number_input('Loan Amount', min_value=0, value=0)
        loan_term = st.number_input('Loan Term (Months)', min_value=0, value=0)
        cibil_score = st.number_input('CIBIL Score', min_value=0, value=0)
        residential_assets_value = st.number_input('Residential Assets Value', min_value=0, value=0)
        commercial_assets_value = st.number_input('Commercial Assets Value', min_value=0, value=0)
        luxury_assets_value = st.number_input('Luxury Assets Value', min_value=0, value=0)
        bank_asset_value = st.number_input('Bank Asset Value', min_value=0, value=0)

        submit_button = st.form_submit_button("Predict")

        if submit_button:
            input_data = {
                'no_of_dependents': no_of_dependents,
                'education': education,
                'self_employed': self_employed,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'residential_assets_value': residential_assets_value,
                'commercial_assets_value': commercial_assets_value,
                'luxury_assets_value': luxury_assets_value,
                'bank_asset_value': bank_asset_value
            }

            input_df = pd.DataFrame([input_data])
            input_df['education'] = input_df['education'].map({'Graduate': 1, 'Not Graduate': 0})
            input_df['self_employed'] = input_df['self_employed'].map({'Yes': 1, 'No': 0})
            input_df = input_df[['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                                 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]

            # Predict without scaling
            prediction = model.predict(input_df)
            result = 'Approved' if prediction[0] == 1 else 'Rejected'

            st.write(f"Loan Status: {result}")

if __name__ == "__main__":
    main()
