import streamlit as st
import cv2
import numpy as np
from main import process_frame
from main_easyocr import process_frame as process_frame_easy

st.title('Document Extraction and Classification')
st.set_page_config(layout="wide")

# File upload section
st.subheader("Upload a picture")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

st.session_state["ocr_mode"] = st.selectbox(
    "Select OCR Mode",
    ("Tesseract OCR", "EasyOCR")
)


if uploaded_file is not None:
    # Convert the uploaded file to a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Process the image
    if st.session_state["ocr_mode"] == "Tesseract OCR":
        processed_img, gray, blurred, features, category = process_frame(img)
    else:
        processed_img, gray, blurred, features, category = process_frame_easy(img)


    st.subheader("Classified Category")
    st.write(category)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        st.subheader("Blurred Image")
        st.image(blurred, channels="GRAY")
    with col2:
        st.subheader("Grayscale Image")
        st.image(gray, channels="GRAY")
    with col3:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), channels="RGB")




