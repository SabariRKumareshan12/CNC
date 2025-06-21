import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import base64
def set_image_local(image_path):
    with open(image_path, "rb") as file:
        img = file.read()
    base64_image = base64.b64encode(img).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            #background-position: center;
            #background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_image_local(r"D:\streamlit\env\CNC\img2.jpg")

# Load Trained LSTM Model
lstm_model = load_model("D:/streamlit/env/CNC/cnc_model.h5")

# Title
st.title("CNC Milling Performance Analysis & Fault Detection")

# Feature Columns (13 Features)
feature_columns = [
    "feedrate", "clamp_pressure", "material",
    "M1_CURRENT_FEEDRATE", "X1_ActualPosition", "Y1_ActualPosition",
    "Z1_ActualPosition", "X1_CurrentFeedback", "Y1_CurrentFeedback",
    "X1_DCBusVoltage", "X1_OutputPower", "Y1_OutputPower_transformed",
    "S1_OutputPower"
]

# Select Input Method: Manual Entry or CSV Upload
input_type = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

if input_type == "Manual Entry":
    # Manual Input for 13 Features
    feedrate = st.number_input("Feedrate (mm/s)", min_value=0.0, step=0.1)
    clamp_pressure = st.number_input("Clamp Pressure (bar)", min_value=0.0, step=0.1)

    # Material Selection (Categorical Encoding)
    material = st.selectbox("Material", ["Wax"])
    material_dict = {"Wax": 0}
    material_encoded = material_dict.get(material, -1)  # Default to -1 if unknown

    M1_CURRENT_FEEDRATE = st.number_input("M1_CURRENT_FEEDRATE", min_value=0.0, step=0.1)
    X1_ActualPosition = st.number_input("X1_ActualPosition", step=0.1)
    Y1_ActualPosition = st.number_input("Y1_ActualPosition", step=0.1)
    Z1_ActualPosition = st.number_input("Z1_ActualPosition", step=0.1)
    X1_CurrentFeedback = st.number_input("X1_CurrentFeedback", step=0.1)
    Y1_CurrentFeedback = st.number_input("Y1_CurrentFeedback", step=0.1)
    X1_DCBusVoltage = st.number_input("X1_DCBusVoltage", step=0.1)
    X1_OutputPower = st.number_input("X1_OutputPower", step=0.1)
    Y1_OutputPower_transformed = st.number_input("Y1_OutputPower_transformed", step=0.1)
    S1_OutputPower = st.number_input("S1_OutputPower", step=0.1)

    if st.button("Predict"):
        # Prepare Input as a NumPy Array
        input_data = np.array([[feedrate, clamp_pressure, material_encoded,
                                M1_CURRENT_FEEDRATE, X1_ActualPosition, Y1_ActualPosition,
                                Z1_ActualPosition, X1_CurrentFeedback, Y1_CurrentFeedback,
                                X1_DCBusVoltage, X1_OutputPower, Y1_OutputPower_transformed,
                                S1_OutputPower]])
        
        # Reshape Input for LSTM (1 sample, 1 timestep, 13 features)
        input_data = input_data.reshape((1, 1, len(feature_columns)))

        # Make Predictions
        prediction = lstm_model.predict(input_data)

        # Display Predictions
        st.write(f"Tool Wear: {'Worn' if prediction[0][0] > 0.5 else 'Unworn'}")
        st.write(f"Clamping Detection: {'Properly Clamped' if prediction[0][1] > 0.5 else 'Not Properly Clamped'}")
        st.write(f"Machining Completion: {'Completed' if prediction[0][2] > 0.5 else 'Not Completed'}")

elif input_type == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CNC Sensor Data (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check for missing columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing Columns in Uploaded File: {missing_cols}")
        else:
            # Convert "material" to numerical encoding
            df["material"] = df["material"].map({"Wax": 0}).fillna('0')

            # Scale Data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[feature_columns])

            # Reshape for LSTM (samples, timesteps, features)
            df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, len(feature_columns)))

            if st.button("Analyze"):
                predictions = lstm_model.predict(df_reshaped)

                # Assign Predictions to DataFrame
                df["Tool Wear"] = ["Worn" if p[0] > 0.5 else "Unworn" for p in predictions]
                df["Clamping Detection"] = ["Properly Clamped" if p[1] > 0.5 else "Not Properly Clamped" for p in predictions]
                df["Machining Completion"] = ["Completed" if p[2] > 0.5 else "Not Completed" for p in predictions]

                # Display and Download Results
                st.write(df)
                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
