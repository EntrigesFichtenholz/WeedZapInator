import os
import json
import yaml
from ultralytics import YOLO

# Verzeichnispfad für JSON-Dateien und Bilder
data_dir = './Trainingsdata'  # Verzeichnis, in dem sich JSON- und Bilddateien befinden
output_model_path = './model.keras'
yaml_file_path = './temp_yolo_data.yaml'  # Pfad zur temporären YAML-Datei

# Funktion zum Laden der JSON-Annotations und Erstellen einer YOLO-YAML-Datei
def create_yolo_yaml(data_dir, yaml_file_path):
    images = []
    annotations = []

    # Alle JSON-Dateien durchlaufen und absolute Pfade sammeln
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            json_path = os.path.join(data_dir, file_name)
            image_path = os.path.join(data_dir, file_name.replace('.json', '.jpg'))  # Bildformat anpassen

            # Überprüfen, ob das zugehörige Bild existiert
            if os.path.exists(image_path):
                images.append(os.path.abspath(image_path))  # Absoluten Pfad hinzufügen
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Annotations- und Klasseninformationen sammeln
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']

                    # Bounding Box berechnen
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Berechne Zentrum, Breite und Höhe für YOLO
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    annotations.append([label, x_center, y_center, width, height])

    # YAML-Datenstruktur für YOLO-Training
    yaml_data = {
        'train': images,
        'val': images[:int(0.2 * len(images))],  # 20% der Bilder als Validierungsdaten
        'nc': len(set([ann[0] for ann in annotations])),  # Anzahl der Klassen
        'names': list(set([ann[0] for ann in annotations]))  # Klassenliste
    }

    # YAML-Datei speichern
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    print(f"YAML-Datei für YOLO erstellt unter: {yaml_file_path}")

# Funktion zum Trainieren des YOLOv8-Modells
def train_yolo_model(yaml_file_path):
    model = YOLO("yolov8n.yaml")  # YOLOv8-Modell initialisieren

    # YOLOv8 Training mit YAML-Datei
    model.train(data=yaml_file_path, epochs=100, batch=16, save=True)

    # Modell speichern
    model.save(output_model_path)

# Hauptprogramm
if __name__ == '__main__':
    # YAML-Datei für YOLO erstellen
    create_yolo_yaml(data_dir, yaml_file_path)

    # Training starten
    print("Training wird gestartet...")
    train_yolo_model(yaml_file_path)
    print(f"Training abgeschlossen. Modell gespeichert unter: {output_model_path}")
