import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
from io import BytesIO

# Load the model
model = tf.keras.models.load_model(r'DiabeticModel.keras')

# Define class labels
class_labels = ['Healthy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

# Function to preprocess the uploaded image
def preprocess_image(image: Image.Image):
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to create PDF report
def create_pdf(patient_name, patient_age, predicted_class, prediction_percentages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diabetic Retinopathy Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Patient Age: {patient_age}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Predicted Level: {predicted_class}", ln=True, align='L')
    
    # Add the DR level prediction details
    for i, label in enumerate(class_labels):
        pdf.cell(200, 10, txt=f"{label}: {prediction_percentages[i]:.2f}%", ln=True, align='L')
    
    # Generate PDF content as a byte string (dest='S' for return as string)
    pdf_content = pdf.output(dest='S').encode('latin1')  # Encoding to 'latin1' ensures compatibility with PDF format
    
    # Convert the byte string to BytesIO object
    pdf_output = BytesIO(pdf_content)
    return pdf_output

# Streamlit interface
st.title("Diabetic Retinopathy Detection App")
st.write("Welcome to our Diabetic Retinopathy Detection App! This app utilizes deep learning models to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.")

# Create tabs for image upload and camera input
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab1:
    # Patient details input
    st.write("### Enter Patient Details")
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Patient Age", min_value=0)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Prepare inputs for model
        img_array_pair = [img_array, img_array]  # Model expects two inputs
        
        # Make prediction
        predictions = model.predict(img_array_pair)[0]
        
        # Convert predictions to percentages
        prediction_percentages = predictions * 100
        
        # Find the class with the highest probability
        highest_index = np.argmax(prediction_percentages)
        predicted_class = class_labels[highest_index]
        
        # Display the image and predictions side by side
        col1, col2 = st.columns([1, 2])  # Set the width ratio between columns
        
        # Display image in the first column with limited width
        with col1:
            st.image(image, caption='Uploaded Image', width=150)
        
        # Display the predictions in the second column
        with col2:
            st.write(f"### Predicted Level: **{predicted_class}**")
            
            st.write("### Prediction Results")
            for i, label in enumerate(class_labels):
                st.progress(int(prediction_percentages[i]))
                st.write(f"{label}: {prediction_percentages[i]:.2f}%")
        
        # Create and download PDF report
        pdf_output = create_pdf(patient_name, patient_age, predicted_class, prediction_percentages)
        
        # Button to download the PDF
        st.download_button(
            label="Download Report",
            data=pdf_output,
            file_name=f"Diabetic_Retinopathy_Report_{patient_name.replace(' ', '_')}.pdf",
            mime='application/octet-stream'
        )

with tab2:
    st.write("### Capture an image using your camera")
    
    # Capture image from camera
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Open and display the captured image
        image = Image.open(camera_image)
        
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Prepare inputs for model
        img_array_pair = [img_array, img_array]  # Model expects two inputs
        
        # Make prediction
        predictions = model.predict(img_array_pair)[0]
        
        # Convert predictions to percentages
        prediction_percentages = predictions * 100
        
        # Find the class with the highest probability
        highest_index = np.argmax(prediction_percentages)
        predicted_class = class_labels[highest_index]
        
        # Display the image and predictions
        col1, col2 = st.columns([1, 2])  # Set the width ratio between columns
        
        # Display image in the first column
        with col1:
            st.image(image, caption='Captured Image', width=150)
        
        # Display the predictions in the second column
        with col2:
            st.write(f"### Predicted Level: **{predicted_class}**")
            
            st.write("### Prediction Results")
            for i, label in enumerate(class_labels):
                st.progress(int(prediction_percentages[i]))
                st.write(f"{label}: {prediction_percentages[i]:.2f}%")
        
        # Create and download PDF report
        pdf_output = create_pdf(patient_name, patient_age, predicted_class, prediction_percentages)
        
        # Button to download the PDF
        st.download_button(
            label="Download Report",
            data=pdf_output,
            file_name=f"Diabetic_Retinopathy_Report_{patient_name.replace(' ', '_')}.pdf",
            mime='application/octet-stream'
        )
