import os
import cv2
import random
import numpy as np
import concurrent.futures

def generate_shapes_with_bounding_boxes(image, num_shapes=10):
    """
    Generiert zufällig Kreise, Rechtecke und Dreiecke auf einem Bild
    und gibt die Bounding-Box-Koordinaten zurück.
    """
    h, w, _ = image.shape
    bounding_boxes = []

    for _ in range(num_shapes):
        # Zufällige Farbauswahl mit Transparenz
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(["circle", "rectangle", "triangle"])

        if shape_type == "circle":
            # Zufällige Position und Radius
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, min(w, h) // 10)

            # Kreis mit Füllung zeichnen
            cv2.circle(image, center, radius, color, thickness=-1)

            x_min = max(0, center[0] - radius)
            y_min = max(0, center[1] - radius)
            x_max = min(w, center[0] + radius)
            y_max = min(h, center[1] + radius)
            class_id = 0  # Class ID für Kreise

        elif shape_type == "rectangle":
            # Zufällige Rechteck-Punkte
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(x1, w), random.randint(y1, h)

            # Rechteck mit Füllung zeichnen
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)

            x_min, y_min, x_max, y_max = x1, y1, x2, y2
            class_id = 1  # Class ID für Rechtecke

        elif shape_type == "triangle":
            # Zufällige Dreiecks-Punkte
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            pt3 = (random.randint(0, w), random.randint(0, h))
            pts = np.array([pt1, pt2, pt3], np.int32)

            # Dreieck mit Füllung zeichnen
            cv2.fillPoly(image, [pts], color)

            x_min = max(0, min(pt1[0], pt2[0], pt3[0]))
            y_min = max(0, min(pt1[1], pt2[1], pt3[1]))
            x_max = min(w, max(pt1[0], pt2[0], pt3[0]))
            y_max = min(h, max(pt1[1], pt2[1], pt3[1]))
            class_id = 2  # Class ID für Dreiecke

        # Normalisierte Koordinaten berechnen
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h

        # Bounding Box speichern
        bounding_boxes.append((class_id, x_center, y_center, bbox_width, bbox_height))

    return image, bounding_boxes

def process_single_image(input_path, output_folder):
    """
    Verarbeitet ein einzelnes Bild mit zufälligen Formen
    """
    filename = os.path.basename(input_path)
    output_image_path = os.path.join(output_folder, filename)
    output_txt_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

    # Bild laden
    image = cv2.imread(input_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {input_path}")
        return

    # Zufällige Formen und Bounding Boxes generieren
    annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes(image)

    # Bild speichern
    cv2.imwrite(output_image_path, annotated_image)

    # Bounding Boxes speichern
    with open(output_txt_path, "w") as f:
        for box in bounding_boxes:
            f.write(" ".join(map(str, box)) + "\n")

    print(f"Verarbeitet: {filename}")

def process_images(input_folder, output_folder, max_workers=None):
    """
    Liest Bilder aus dem Eingabeordner, generiert zufällige Formen und Bounding-Box-Daten
    und speichert die Ergebnisse im Ausgabeordner mit Multithreading
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Alle Bildpfade sammeln
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Multithreading mit ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Erstelle eine Liste von Futures
        futures = [
            executor.submit(process_single_image, image_path, output_folder)
            for image_path in image_paths
        ]

        # Warte auf Abschluss aller Aufgaben
        concurrent.futures.wait(futures)

# Hauptfunktion
if __name__ == "__main__":
    input_folder = "Gen_input_images"  # Ordner mit den Eingabebildern
    output_folder = "Gen_output_images"  # Ordner für die Ergebnisse

    # Optional: Anzahl der Worker festlegen (None für automatische Auswahl)
    process_images(input_folder, output_folder, max_workers=None)
