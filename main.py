import cv2
from CardAnalyser import CardAnalyzer 

# --- EXECUTION ---

analyzer = CardAnalyzer()


# Run the live video
# run_video_mode()

analyzer.run_image_mode('ressources\carteetudiant2.jpg')
# analyzer.run_image_mode('ressources\carteidentite5.jpg')
# analyzer.run_image_mode('ressources\cartefidelite5.jpg')


