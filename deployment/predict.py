import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

def run():
    # Load model
    with open('bestmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Title
    st.title('Patient Show Up Predictor')

    # Input banner
    st.image('https://hybrid.co.id/wp-content/uploads/2022/02/286002f850374c51cc10f6924e40a1dd_FIFA-22.jpeg')

    # Description
    st.write(''' This page will allow users to predict whether a patient will show up based on their appointment data''')

    # Form for input
    with st.form(key='form_parameter'):
        patientid = st.text_input('Patient ID:', 'Input')
        appointmentid = st.text_input('Appointment ID:', 'Input')
        gender = st.selectbox('Gender', ('M', 'F'))
        age = st.number_input('Age of Patient:', min_value=1, step=1)

        # Input for Scheduled Day and Appointment Day
        scheduled_day = st.date_input('Scheduled Day:', value=datetime.today())
        appointment_day = st.date_input('Appointment Day:', value=datetime.today())

        neighbourhood = st.text_input('Neighbourhood:', 'Input')
        scholarship = st.checkbox('Scholarship')
        hipertension = st.checkbox('Hypertension')
        diabetes = st.checkbox('Diabetes')
        alcoholism = st.checkbox('Alcoholism')
        handcap = st.checkbox('Handicap')
        sms_received = st.checkbox('SMS received')

        submit = st.form_submit_button('Predict Show Up')

    # Data Inference
    if submit:
        # Calculate the difference in days
        date_diff = (appointment_day - scheduled_day).days

        df = pd.DataFrame([{
            'PatientId': float(patientid),  # Convert to float for consistency
            'AppointmentID': int(appointmentid),  # Convert to int for consistency
            'Gender': gender,
            'ScheduledDay': scheduled_day.strftime('%Y-%m-%d'),  # Format as string
            'AppointmentDay': appointment_day.strftime('%Y-%m-%d'),  # Format as string
            'Age': age,
            'Neighbourhood': neighbourhood,
            'Scholarship': scholarship,
            'Hipertension': hipertension,
            'Diabetes': diabetes,
            'Alcoholism': alcoholism,
            'Handcap': handcap,
            'SMS_received': sms_received,
            'Showed_up': None,  # Not used for prediction, set to None
            'Date.diff': date_diff  # Use calculated date difference
        }])

        # Predict
        pred = model.predict(df)
        st.write(f"#### Predicted Show Up: {pred[0]}")

if __name__ == '__main__':
    run()
