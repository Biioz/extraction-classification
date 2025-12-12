import streamlit as st
import cv2
import numpy as np
from main import process_frame

st.title('Extraction et Classification de Documents')
st.set_page_config(layout="wide")

# Section de téléchargement de fichier
st.subheader("Télécharger une image")
uploaded_file = st.file_uploader("Choisir un fichier", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convertir le fichier téléchargé en tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Traiter l'image
    processed_img, gray, blurred, features, category, scores = process_frame(img)


    st.subheader("Catégorie Classifiée")
    st.write(category)
    st.subheader("Scores")
    st.write(scores)

    # Afficher les résultats
    col1, col2, col3 = st.columns(3)


    with col1:
        st.subheader("Image Floutée")
        st.image(blurred, channels="GRAY")
    with col2:
        st.subheader("Image en Niveaux de Gris")
        st.image(gray, channels="GRAY")
    with col3:
        st.subheader("Image Traitée")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), channels="RGB")




