import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import time
from main import process_frame

# Variables globales pour le calcul du FPS
frame_count = 0
start_time = time.time()
fps = 0


def add_label(img, text, position='top'):
    """Ajoute un label sur l'image"""
    h, w = img.shape[:2]
    label_height = 40
    label_bar = np.zeros((label_height, w, 3), dtype=np.uint8)
    
    cv2.putText(label_bar, text, (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if position == 'top':
        return np.vstack([label_bar, img])
    else:
        return np.vstack([img, label_bar])

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Traite la frame et retourne les 3 images combinées"""
    global frame_count, start_time, fps
    
    img = frame.to_ndarray(format="bgr24")
    
    # Calculer le FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 1.0:  # Mettre à jour le FPS chaque seconde
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # Traiter l'image
    processed_img, gray, blurred, features, category = process_frame(img)
    
    # Convertir gray en BGR si nécessaire
    if len(gray.shape) == 2:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray_bgr = gray
    
    # Convertir blurred en BGR si nécessaire
    if len(blurred.shape) == 2:
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    else:
        blurred_bgr = blurred
    
    # Redimensionner les images pour qu'elles aient la même hauteur
    h = processed_img.shape[0]
    gray_resized = cv2.resize(gray_bgr, (gray_bgr.shape[1], h))
    blurred_resized = cv2.resize(blurred_bgr, (blurred_bgr.shape[1], h))
    
    # Ajouter des labels
    
    img1 = add_label(processed_img, "Image Traitee")
    img2 = add_label(gray_resized, "Niveaux de Gris")
    img3 = add_label(blurred_resized, "Image Floutee")
    
    # Combiner horizontalement
    combined = np.hstack([img1, img2, img3])
    # combined = np.hstack([img1])
    
    # Ajouter les informations en bas
    info_height = 80
    info_bar = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)
    
    # Afficher le FPS en haut à droite
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(combined, fps_text, (combined.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Afficher les caractéristiques
    if features is not None and isinstance(features, dict):
        text_detected = features.get('text', 'Aucun')
        has_face = features.get('has_face', False)
        face_status = "Visage: OUI" if has_face else "Visage: NON"
        
        cv2.putText(info_bar, f"Texte: {text_detected[:30]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_bar, face_status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if category is not None:
        cv2.putText(info_bar, f"Categorie: {category}", (10 + combined.shape[1]//2, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Combiner verticalement avec la barre d'info
    final_output = np.vstack([combined, info_bar])
    
    return av.VideoFrame.from_ndarray(final_output, format="bgr24")

# Configuration
st.set_page_config(page_title="Analyse Vidéo", layout="wide")

st.title("🎥 Analyse Vidéo en Temps Réel")

st.info("📺 Le flux vidéo affiche 3 vues en temps réel : Image Traitée | Niveaux de Gris | Image Floutée")

# WebRTC Streamer unique avec tout combiné
webrtc_streamer(
    key="combined-analysis",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.markdown("---")
st.markdown("""
### 📋 Légende
- **Image Traitée** : Image originale avec détections
- **Niveaux de Gris** : Conversion en nuances de gris
- **Image Floutée** : Application de flou gaussien
- **Barre du bas** : Informations de détection (texte, visage, catégorie)
- **FPS** : Affiché en haut à droite du flux vidéo
""")