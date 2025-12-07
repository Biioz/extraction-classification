import cv2
import numpy as np
import matplotlib.pyplot as plt

def Preprocessing(image_path):
    # 1. Chargement de l'image
    img = cv2.imread(image_path)
    if img is None:
        print("Erreur de chargement")
        return None

    # 2. Conversion en niveaux de gris (Base indispensable)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- ÉTAPE A : RÉDUCTION DU BRUIT (Selon Expérience 5) ---
    
    # Option 1 : Filtre Médian (Idéal pour bruit "Poivre et Sel" / Poussières)
    # Le paramètre '5' est la taille du noyau (doit être impair).
    # Cela remplace chaque pixel par la médiane de ses voisins.
    # denoised_median = cv2.medianBlur(gray, 3)

    # Option 2 : Filtre Bilatéral (Idéal pour bruit Gaussien / Grain photo)
    # C'est souvent MIEUX que le flou Gaussien simple pour du texte, 
    # car il préserve les arêtes (les bords des lettres) tout en lissant le fond.
    # d=9: diamètre voisinage, sigmaColor=75: mélange couleurs, sigmaSpace=75: mélange espace
    denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # --- ÉTAPE B : AUGMENTATION DU CONTRASTE / SEUILLAGE ---

    # Le seuillage adaptatif est crucial pour les cartes avec des motifs de fond.
    # Il calcule le seuil pour chaque petite zone, ce qui gère les ombres ou les fonds dégradés.
    # Block Size (11) et C (2) sont à ajuster.
    thresh = cv2.adaptiveThreshold(
        denoised_bilateral,         # On utilise l'image débruitée
        255,                        # Valeur max
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Méthode de calcul
        cv2.THRESH_BINARY,          # Type de seuillage (Noir/Blanc)
        11,                         # Taille du bloc de voisinage
        2                           # Constante soustraite (Nettoyage fin)
    )

    # --- ÉTAPE C : NETTOYAGE MORPHOLOGIQUE (Optionnel) ---
    # Parfois le seuillage laisse des petits points blancs (bruit restant).
    # Une "Ouverture" (Erosion puis Dilatation) les supprime.
    kernel = np.ones((2,2), np.uint8) # Petit noyau
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening

