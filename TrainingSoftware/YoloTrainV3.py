"""from ultralytics import YOLO

model= YOLO("yolo11x.pt")

results = model.train(data="Configv5.yaml", epochs=1)
"""
import datetime
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="Configv5.yaml",
    epochs=100,
    device=[0, 1, 2, 3],   # Nutzt beide GPUs
    batch=8,         # Kleinere Batchgröße
    amp=True,        # Automatische gemischte Präzision (half precision)
    cache=True,
    imgsz=720        # Reduzierte Bildgröße
)

# Erstelle den "savemodels" Ordner, falls er noch nicht existiert
Path("savemodels").mkdir(parents=True, exist_ok=True)

# Hole den aktuellen Datum und Zeit
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Speichern des Modells
model.save(f"savemodels/mein_modell_{timestamp}.pt")

# Speichern der Trainingsergebnisse
results.save(f"savemodels/train_results_{timestamp}.yaml")
