"""from ultralytics import YOLO

model= YOLO("yolo11x.pt")

results = model.train(data="Configv5Shapes.yaml", epochs=1)
"""
import datetime
from pathlib import Path
from ultralytics import YOLO

model = YOLO("./savemodels/302ZyklenModel_20241215_024611.pt")
results = model.train(
    data="Configv5Shapes.yaml",
    epochs=50,
    device=[0, 1, 2, 3],   # Nutzt beide GPUs
    batch=32,         # Kleinere Batchgröße
    amp=True,        # Automatische gemischte Präzision (half precision)
    cache=True,
    imgsz=1080,        # Reduzierte Bildgröße
)

# Erstelle den "savemodels" Ordner, falls er noch nicht existiert
Path("savemodels").mkdir(parents=True, exist_ok=True)

# Hole den aktuellen Datum und Zeit
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Speichern des Modells
model.save(f"savemodels/MultiRes302ZyklenModel_20241215_024611_{timestamp}.pt")

# Speichern der Trainingsergebnisse
results.save(f"savemodels/MultiRes302ZyklenModel_20241215_024611_{timestamp}.yaml")
