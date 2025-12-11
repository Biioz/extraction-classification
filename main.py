import cv2
from CardAnalyser import CardAnalyzer 

# --- EXECUTION ---

analyzer = CardAnalyzer()

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
        features = analyzer.extract_features(frame, blurred)
        category = analyzer.classify_document(features)
        
        # Draw
        result_frame = analyzer.draw_results(frame, features, category)
        
        cv2.imshow("Project Video Analysis", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Vérifier si la fenêtre a été fermée
        if cv2.getWindowProperty("Project Video Analysis", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
def process_frame(frame):
    """
    Analyse un frame et retourne un frame annoté.
    Aucun appel Streamlit ou imshow ici.
    """
    gray, blurred = analyzer.preprocess_image(frame)
    features = analyzer.extract_features(frame, blurred)
    category = analyzer.classify_document(features)
    result_frame = analyzer.draw_results(frame, features, category)

    return result_frame , gray, blurred, features, category
    

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

# analyzer.run_image_mode('ressources\carteetudiant5.jpg')
# analyzer.run_image_mode('ressources\carteidentite5.jpg')
# analyzer.run_image_mode('ressources\cartefidelite5.jpg')


