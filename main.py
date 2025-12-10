import CardAnalyser
import preprocessing    
import os
import json

# --- UTILISATION ---
if __name__ == "__main__":
    
    image_input = "ressources\cartefidelite2.jpg" 
    
    if os.path.exists(image_input):

        preprocessed_image = preprocessing.Preprocessing(image_input)
        
        analyzer =  CardAnalyser.CardAnalyser(image_input, use_gpu=False) 
        final_data = analyzer.run()
        
        # Sauvegarder les données JSON
        with open("output/resultats.json", "w", encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        
        # Générer l'image visuelle
        analyzer.draw_results()

        print("\n--- ANALYSE TERMINÉE ---")
        print("Noms potentiels :", final_data["extracted_data"]["possible_names"])
        print("Types de carte détectés :", final_data["extracted_data"]["type_carte"])
        print("Consultez 'resultats.json' pour le détail complet.")
        
    else:
        print(f"Erreur : L'image '{image_input}' n'existe pas.")