import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from main import process_frame

# --- 1. Fonction de rappel (Callback Function) ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convertir le frame AV en tableau NumPy (format BGR pour OpenCV)
    img = frame.to_ndarray(format="bgr24")

    # --- 2. Votre Logique d'Analyse en Temps Réel ---
    # Exemple : Convertir l'image en niveaux de gris
    
    processed_img = process_frame(img)
    # *Si vous faites de la détection d'objets (YOLO/Mediapipe/etc.), c'est ici que vous dessinez les boîtes englobantes.*
    print("Processing frame...")
    print("Frame shape:", processed_img.shape)

    # --- 3. Retourner le frame traité ---
    # Convertir le tableau NumPy traité en frame AV
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- 4. Intégration Streamlit ---
st.title("Analyse Vidéo en Temps Réel avec Streamlit")

# Le composant webrtc_streamer démarre le flux de la webcam
webrtc_streamer(
    key="real-time-analysis", # Clé unique pour le composant
    video_frame_callback=video_frame_callback, # Votre fonction d'analyse
    media_stream_constraints={"video": True, "audio": False}, # Activer la vidéo, désactiver l'audio
    async_processing=True # Permet d'éviter de bloquer le thread principal de Streamlit
)