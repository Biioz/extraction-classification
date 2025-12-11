import cv2
import pytesseract
from pytesseract import Output # Import Output to get dictionary format

# CONFIGURATION
# path to point to your tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentAnalyzer:
    def __init__(self):
        # Load the Face Detector
        # Make sure 'haarcascade_frontalface_default.xml' is in the same folder
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # MODULE 1: PREPROCESSING
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        return gray_resized, blurred

    # MODULE 2: SEGMENTATION & EXTRACTION
    def extract_features(self, image, gray_image):
        features = {
            "text": "",
            "text_coords": [], # List to store text bounding boxes
            "has_face": False,
            "face_coords": []
        }

        # 1. Text Extraction (Updated to get coordinates)
        # output_type=Output.DICT gives us a dictionary with 'left', 'top', 'width', 'height', 'text'
        data = pytesseract.image_to_data(gray_image, output_type=Output.DICT)
        
        full_text_list = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Filter out empty text and low confidence results (noise)
            if int(data['conf'][i]) > 40: 
                word = data['text'][i].strip()
                if word:
                    # Store the word and its coordinates
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    features["text_coords"].append((x, y, w, h))
                    full_text_list.append(word)

        # Join all words to make the full text string for classification
        features["text"] = " ".join(full_text_list).lower()

        # 2. Face Detection
        faces = self.face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            features["has_face"] = True
            features["face_coords"] = faces

        return features

    # MODULE 3: CLASSIFICATION
    def classify_document(self, features):
        text = features["text"]
        has_face = features["has_face"]
        
        id_keywords = ["republique", "française", "identite","d'identite", "national"]
        student_keywords = ["etudiant", "universite", "ecole", "ine"]
        loyalty_keywords = ["fidelite", "points", "magasin", "client"]

        scores = {"ID": 0, "STUDENT": 0, "LOYALTY": 0}

        # Keyword Matching
        for word in id_keywords:
            if word in text: 
                scores["ID"] += 1
                print("+1 id for", word)
        for word in student_keywords:
            if word in text: 
                scores["STUDENT"] += 1
                print("+1 student for", word)
        for word in loyalty_keywords:
            if word in text: 
                scores["LOYALTY"] += 1
                print("+1 loyalty for", word)

        # Visual Heuristics
        if has_face:
            scores["ID"] += 2
            scores["STUDENT"] += 2
        else:
            scores["LOYALTY"] += 2

        best_match = max(scores, key=scores.get)
        if scores[best_match] == 0:
            return "UNKNOWN"
        return best_match

    # VISUALIZATION
    def draw_results(self, image, features, category):
        output = image.copy()
        
        # 1. Draw Text Rectangles (Yellow)
        for (x, y, w, h) in features["text_coords"]:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # 2. Draw Face Rectangles (Blue)
        for (x, y, w, h) in features["face_coords"]:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 3. Draw Classification Label (Green)
        cv2.putText(output, f"Type: {category}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output

# --- EXECUTION ---

analyzer = DocumentAnalyzer()

def run_video_mode():
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Video Mode. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Process frame
        gray, blurred = analyzer.preprocess_image(frame)
        features = analyzer.extract_features(frame, gray)
        category = analyzer.classify_document(features)
        
        # Draw
        result_frame = analyzer.draw_results(frame, features, category)
        
        cv2.imshow("Project Video Analysis", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_image_mode(image_path):
    print(f"--- Processing {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Pipeline
    gray, blurred = analyzer.preprocess_image(img)
    features = analyzer.extract_features(img, blurred)
    category = analyzer.classify_document(features)
    
    print(f"Detected Text Snippet: {features['text'][:50]}...")
    print(f"Face Detected: {features['has_face']}")
    print(f"Result: {category}")

    # Show result
    result_img = analyzer.draw_results(img, features, category)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the live video
# run_video_mode()

run_image_mode('ressources\carteetudiant2.jpg')
# run_image_mode('ressources\carteidentite5.jpg')
# run_image_mode('ressources\cartefidelite5.jpg')


