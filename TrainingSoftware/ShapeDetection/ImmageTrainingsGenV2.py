import os
import cv2
import random
import numpy as np
import concurrent.futures

def add_shadow(image):
    """
    Fügt Schatteneffekte durch zufällige transparente Formen hinzu.
    """
    h, w, _ = image.shape
    overlay = image.copy()
    for _ in range(random.randint(3, 6)):
        shape_type = random.choice(["circle", "rectangle"])
        alpha = random.uniform(0.3, 0.7)  # Transparenz
        color = (0, 0, 0)  # Schwarze Schattenfarbe

        if shape_type == "circle":
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(30, min(w, h) // 4)
            cv2.circle(overlay, center, radius, color, thickness=-1)

        elif shape_type == "rectangle":
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(x1, w), random.randint(y1, h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def add_magnifying_glass_effect(image):
    """
    Fügt einen Lupeneffekt hinzu.
    """
    h, w, _ = image.shape
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Linsenzentrum und Stärke zufällig wählen
    lens_center = (random.randint(0, w), random.randint(0, h))
    lens_radius = random.randint(50, min(w, h) // 4)
    lens_strength = random.uniform(1.1, 1.5)  # Verstärkung der Verzerrung

    # Erstelle eine Linsenmaske
    cv2.circle(mask, lens_center, lens_radius, 255, -1)

    # Linsenverzerrung anwenden
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                dx = x - lens_center[0]
                dy = y - lens_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < lens_radius:
                    scale = lens_strength - (lens_strength - 1) * (distance / lens_radius)
                    nx = int(lens_center[0] + dx / scale)
                    ny = int(lens_center[1] + dy / scale)
                    if 0 <= nx < w and 0 <= ny < h:
                        overlay[y, x] = image[ny, nx]
                    else:
                        overlay[y, x] = image[y, x]
            else:
                overlay[y, x] = image[y, x]

    return overlay

def add_noise(image):
    """
    Fügt zufälliges Rauschen zum Bild hinzu.
    """
    noise = np.random.normal(0, 25, image.shape).astype(np.int32)
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8U)
    return noisy_image

def generate_shapes_with_bounding_boxes(image, num_shapes=10):
    """
    Generiert zufällig Kreise, Rechtecke und Dreiecke auf einem Bild
    und gibt die Bounding-Box-Koordinaten zurück.
    """
    h, w, _ = image.shape
    bounding_boxes = []

    for _ in range(num_shapes):
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(["circle", "rectangle", "triangle"])

        if shape_type == "circle":
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, min(w, h) // 10)
            cv2.circle(image, center, radius, color, thickness=-1)
            x_min = max(0, center[0] - radius)
            y_min = max(0, center[1] - radius)
            x_max = min(w, center[0] + radius)
            y_max = min(h, center[1] + radius)
            class_id = 0

        elif shape_type == "rectangle":
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(x1, w), random.randint(y1, h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
            x_min, y_min, x_max, y_max = x1, y1, x2, y2
            class_id = 1

        elif shape_type == "triangle":
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            pt3 = (random.randint(0, w), random.randint(0, h))
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.fillPoly(image, [pts], color)
            x_min = max(0, min(pt1[0], pt2[0], pt3[0]))
            y_min = max(0, min(pt1[1], pt2[1], pt3[1]))
            x_max = min(w, max(pt1[0], pt2[0], pt3[0]))
            y_max = min(h, max(pt1[1], pt2[1], pt3[1]))
            class_id = 2

        # Normalisierte Koordinaten berechnen
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h
        bounding_boxes.append((class_id, x_center, y_center, bbox_width, bbox_height))

    return image, bounding_boxes

def process_single_image(input_path, output_folder):
    """
    Verarbeitet ein einzelnes Bild mit verschiedenen Effekten
    """
    filename = os.path.basename(input_path)
    output_image_path = os.path.join(output_folder, filename)
    output_txt_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

    # Bild laden
    image = cv2.imread(input_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {input_path}")
        return

    # Bildeffekte anwenden
    image, bboxes = generate_shapes_with_bounding_boxes(image)
    image = add_shadow(image)
    image = add_magnifying_glass_effect(image)
    image = add_noise(image)

    # Bild speichern
    cv2.imwrite(output_image_path, image)

    # Bounding Boxes speichern
    with open(output_txt_path, "w") as f:
        for box in bboxes:
            f.write(" ".join(map(str, box)) + "\n")

def process_images(input_folder, output_folder, max_workers=None):
    """
    Verarbeitet alle Bilder im Eingabeordner
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Bildpfade sammeln
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_single_image, image_paths, [output_folder]*len(image_paths))

if __name__ == "__main__":
    input_folder = "Gen_input_images"
    output_base = "Gen_output_images"

    # Stelle sicher, dass der Hauptordner existiert
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    # Ordner für Train und Validate
    train_folder = os.path.join(output_base, "train")
    validate_folder = os.path.join(output_base, "validate")

    # Erzeuge Trainings- und Validierungsordner
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validate_folder, exist_ok=True)

    # Verarbeite Bilder für Training
    print("Starte Verarbeitung für Trainingsbilder")
    process_images(input_folder, train_folder)

    # Verarbeite Bilder für Validierung
    print("Starte Verarbeitung für Validierungsbilder")
    process_images(input_folder, validate_folder)
