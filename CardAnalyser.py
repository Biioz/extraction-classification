import cv2
import easyocr
import re
import numpy as np

class CardAnalyser:

    def __init__(self, image_path, use_gpu=False):

        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
        
        print("Chargement du modèle EasyOCR...")
        self.reader = easyocr.Reader(['fr', 'en'], gpu=use_gpu)
        
        self.results = {
            "visual_elements": {},
            "text_analysis": [],
            "extracted_data": {
                "dates": [],
                "mrz": [],
                "possible_names": [],
                "type_carte": []
            }
        }

    def detect_face(self):
        """
        Détection du visage avec OpenCV (Haar Cascade).
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            self.results["visual_elements"]["face"] = {
                "detected": True,
                "bbox": [int(x), int(y), int(w), int(h)]
            }
        else:
            self.results["visual_elements"]["face"] = {"detected": False}

    def process_text(self):
        """
        Extraction du texte et des coordonnées avec EasyOCR.
        """
        # EasyOCR renvoie une liste de tuples : (bbox, text, confidence)
        ocr_results = self.reader.readtext(self.image)
        

        for (bbox, text, prob) in ocr_results:

            # Nettoyage des types pour le JSON (int32 non sérialisable)
            clean_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            entry = {
                "text": text,
                "confidence": float(prob),
                "bbox": clean_bbox # Coordonnées des 4 coins du mot
            }
            self.results["text_analysis"].append(entry)

            # --- Extraction intelligente ---
            
            # 1. Détection de Dates (ex: 12/05/1990 ou 12.05.1990)
            if re.search(r'\d{2}[./]\d{2}[./]\d{4}', text):
                self.results["extracted_data"]["dates"].append(text)

            # # Détection MRZ (lignes avec beaucoup de <<<)
            # if "<<" in text and len(text) > 15:
            #     self.results["extracted_data"]["mrz"].append(text)

            # 2. Détection de noms potentiels (Mots en MAJUSCULES sans chiffres, longueur > 2)
            if text.isalpha() and len(text) > 2:
                self.results["extracted_data"]["possible_names"].append(text)

            if not self.results["extracted_data"]["type_carte"]:
                
                # 3. Détection des cartes d'identité
                if "CARTE NATIONALE DIDENTITÉ" in text:
                    self.results["extracted_data"]["type_carte"].append("Carte d'identité")

                # 4. Détection des cartes de fidélité
                if "FIDELITE" in text or "client" in text or "MA CARTE" in text:
                    self.results["extracted_data"]["type_carte"].append("Carte de fidélité")

                # 5. Détection des cartes étudiantes
                if "ETUDIANT" in text or "INE" in text:
                    self.results["extracted_data"]["type_carte"].append("Carte étudiante")


    def draw_results(self, output_path="output\output_debug.jpg"):
        """
        Génère une image de debug avec les cadres autour du texte et du visage.
        Utile pour visualiser ce que l'IA a "vu".
        """
        img_copy = self.image.copy()
        
        # Dessiner le visage
        face = self.results["visual_elements"].get("face")
        if face and face["detected"]:
            x, y, w, h = face["bbox"]
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Dessiner le texte
        for item in self.results["text_analysis"]:
            # EasyOCR donne 4 points : top_left, top_right, bottom_right, bottom_left
            pts = np.array(item["bbox"], np.int32)
            cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imwrite(output_path, img_copy)
        print(f"Image de debug sauvegardée sous : {output_path}")

    def run(self):
        print("Détection du visage...")
        self.detect_face()
        print("Lecture du texte en cours...")
        self.process_text()
        return self.results