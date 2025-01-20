import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os 

st.header('Car Price Prediction ML Model')

# Car Image Upload
st.header("Upload Car's Image")


uploaded_file = st.file_uploader("Choose a car image to upload", type=["jpg", "jpeg", "png"])

car_image_path = None  

# Display the uploaded image
if uploaded_file is not None:
    
    car_image = Image.open(uploaded_file)
    st.image(car_image, caption="Uploaded Car Image", use_column_width=True)
    st.success("Image successfully uploaded!")
    
    
    car_image_path = os.path.join(os.getcwd(), "uploaded_car_image.jpg")
    car_image.save(car_image_path)
else:
    st.warning("No image uploaded yet. Please upload an image.")

model = pk.load(open('model.pkl', 'rb'))

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Seller  type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)
color = st.selectbox('Car Color', ['White', 'Black', 'Red', 'Other'])

# Car Condition Slider
condition = st.radio("Select Car Condition", ["Excellent", "Good", "Average", "Poor"])

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'], list(range(1, 32)), inplace=True)

    car_price = model.predict(input_data_model)

    # Adjust price based on color
    if color == 'White':
        car_price[0] += 10000
    elif color == 'Black':
        car_price[0] += 20000
    elif color == 'Red':
        car_price[0] += 15000

    # Adjust price based on condition
    condition_multiplier = {
        "Excellent": 1.02,
        "Good": 1.01,
        "Average": 1.0,
        "Poor": 0.9
    }
    adjusted_price = car_price[0] * condition_multiplier[condition]

    st.markdown(f"**Predicted Car Price : ₹{adjusted_price:.2f}**")

    # Generate Prediction Report
    report_buffer = io.BytesIO()
    pdf = canvas.Canvas(report_buffer, pagesize=letter)
    pdf.drawString(100, 750, "Car Price Prediction Report")
    pdf.drawString(100, 730, f"Brand: {name}")
    pdf.drawString(100, 710, f"Year: {year}")
    pdf.drawString(100, 690, f"KMs Driven: {km_driven}")
    pdf.drawString(100, 670, f"Fuel Type: {fuel}")
    pdf.drawString(100, 650, f"Seller Type: {seller_type}")
    pdf.drawString(100, 630, f"Transmission: {transmission}")
    pdf.drawString(100, 610, f"Owner Type: {owner}")
    pdf.drawString(100, 590, f"Mileage: {mileage} kmpl")
    pdf.drawString(100, 570, f"Engine: {engine} CC")
    pdf.drawString(100, 550, f"Max Power: {max_power} bhp")
    pdf.drawString(100, 530, f"Seats: {seats}")
    pdf.drawString(100, 510, f"Color: {color}")
    pdf.drawString(100, 490, f"Condition: {condition}")
    pdf.drawString(100, 470, f"Predicted Price: ₹{adjusted_price:.2f}")

    # Add image to the PDF
    if car_image_path:
        pdf.drawImage(car_image_path, 100, 300, width=200, height=150)  # Adjust position and size as needed

    pdf.save()
    report_buffer.seek(0)

    st.download_button(label="Download Prediction Report", data=report_buffer, file_name="Car_Prediction_Report.pdf", mime="application/pdf")
