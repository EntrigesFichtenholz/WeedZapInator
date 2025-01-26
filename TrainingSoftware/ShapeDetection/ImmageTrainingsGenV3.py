import os
import cv2
import random
import numpy as np
import concurrent.futures
from shapely.geometry import Polygon
def add_complex_lens_effect(image, lens_strength=None, lens_center=None, curve_power=None, oval_factor=None, wave_amplitude=None):
    """
    Fügt einen komplexen Lupeneffekt mit ovalen Formen und leicht gekrümmten Linien hinzu.

    Args:
        image (numpy.ndarray): Eingabebild
        lens_strength (float, optional): Stärke der Linsenverzerrung
        lens_center (tuple, optional): Zentrum der Verzerrung
        curve_power (float, optional): Exponent für zusätzliche Krümmungskontrolle
        oval_factor (float, optional): Verhältnis für anisotrope Verzerrung (Ovale)
        wave_amplitude (float, optional): Amplitude der wellenförmigen Krümmung

    Returns:
        numpy.ndarray: Bild mit komplexem Lupeneffekt
    """
    h, w = image.shape[:2]

    # Standardwerte
    if lens_strength is None:
        lens_strength = random.uniform(0.5, 2.0)
    if lens_center is None:
        lens_center = (random.randint(w // 4, 3 * w // 4),
                       random.randint(h // 4, 3 * h // 4))
    if curve_power is None:
        curve_power = random.uniform(2.0, 4.0)
    if oval_factor is None:
        oval_factor = random.uniform(0.5, 1.5)  # Verändert die X/Y-Skalierung
    if wave_amplitude is None:
        wave_amplitude = random.uniform(5, 20)  # Amplitude der Wellenkrümmung

    # Gitter erstellen
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = (x - lens_center[0]) * oval_factor  # Anisotrope Verzerrung
    dy = y - lens_center[1]
    distance = np.sqrt(dx**2 + dy**2)

    # Verzerrungskurve
    scale = 1 + lens_strength * np.power(np.tanh(distance / 100), curve_power)

    # Wellenförmige Deformation hinzufügen
    wave_x = wave_amplitude * np.sin(2 * np.pi * y / h)  # Horizontale Welle
    wave_y = wave_amplitude * np.cos(2 * np.pi * x / w)  # Vertikale Welle

    # Mapping-Koordinaten
    map_x = (x - lens_center[0] + wave_x) / scale + lens_center[0]
    map_y = (y - lens_center[1] + wave_y) / scale + lens_center[1]

    # Verzerrtes Bild erzeugen
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    return distorted


def add_random_shadow(image):
    """
    Fügt einen zufälligen Schatten zum Bild hinzu.

    Args:
        image (numpy.ndarray): Eingabebild

    Returns:
        numpy.ndarray: Bild mit Schatten
    """
    h, w = image.shape[:2]

    # Zufällige Schattenform
    shadow_type = random.choice(['polygon', 'rectangle'])

    # Zufällige Transparenz
    alpha = random.uniform(0.1, 0.4)

    # Kopie des Bildes erstellen
    shadow_image = image.copy()

    if shadow_type == 'polygon':
        # Zufälliges Polygon
        num_points = random.randint(3, 6)
        points = [(random.randint(0, w), random.randint(0, h)) for _ in range(num_points)]

        # Polygon zeichnen
        overlay = shadow_image.copy()
        cv2.fillPoly(overlay, [np.array(points)], (0, 0, 0))

        # Transparenz hinzufügen
        cv2.addWeighted(overlay, alpha, shadow_image, 1 - alpha, 0, shadow_image)

    else:  # Rechteck
        # Zufälliges Rechteck
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(x1, w), random.randint(y1, h)

        # Rechteck zeichnen
        overlay = shadow_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Transparenz hinzufügen
        cv2.addWeighted(overlay, alpha, shadow_image, 1 - alpha, 0, shadow_image)

    return shadow_image

def add_noise(image):
    """
    Fügt Rauschen zum Bild hinzu.

    Args:
        image (numpy.ndarray): Eingabebild

    Returns:
        numpy.ndarray: Bild mit Rauschen
    """
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def generate_shapes_with_bounding_boxes(image, num_shapes=10):
    """
    Generiert zufällig Kreise, Rechtecke und Dreiecke auf einem Bild
    und gibt die Bounding-Box-Koordinaten zurück.

    Vermeidet vollständige Überlappungen von Objekten.
    """
    h, w, _ = image.shape
    bounding_boxes = []
    existing_polygons = []

    for _ in range(num_shapes):
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(["circle", "rectangle", "triangle"])

        max_attempts = 10
        for attempt in range(max_attempts):
            if shape_type == "circle":
                center = (random.randint(0, w), random.randint(0, h))
                radius = random.randint(10, min(w, h) // 10)

                # Polygon für Kollisionserkennung
                circle_poly = Polygon([(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t))
                                       for t in np.linspace(0, 2*np.pi, 20)])

                # Überlappungsprüfung
                if not any(circle_poly.intersects(existing_poly) for existing_poly in existing_polygons):
                    cv2.circle(image, center, radius, color, thickness=-1)

                    x_min = max(0, center[0] - radius)
                    y_min = max(0, center[1] - radius)
                    x_max = min(w, center[0] + radius)
                    y_max = min(h, center[1] + radius)

                    class_id = 0
                    existing_polygons.append(circle_poly)
                    break

            elif shape_type == "rectangle":
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = random.randint(x1, w), random.randint(y1, h)

                rect_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                if not any(rect_poly.intersects(existing_poly) for existing_poly in existing_polygons):
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)

                    x_min, y_min, x_max, y_max = x1, y1, x2, y2
                    class_id = 1
                    existing_polygons.append(rect_poly)
                    break

            elif shape_type == "triangle":
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                pt3 = (random.randint(0, w), random.randint(0, h))
                pts = np.array([pt1, pt2, pt3], np.int32)

                triangle_poly = Polygon([pt1, pt2, pt3])

                if not any(triangle_poly.intersects(existing_poly) for existing_poly in existing_polygons):
                    cv2.fillPoly(image, [pts], color)

                    x_min = max(0, min(pt1[0], pt2[0], pt3[0]))
                    y_min = max(0, min(pt1[1], pt2[1], pt3[1]))
                    x_max = min(w, max(pt1[0], pt2[0], pt3[0]))
                    y_max = min(h, max(pt1[1], pt2[1], pt3[1]))

                    class_id = 2
                    existing_polygons.append(triangle_poly)
                    break

            if attempt == max_attempts - 1:
                # Keine nicht-überlappende Position gefunden
                continue

        # Nur hinzufügen, wenn erfolgreich platziert
        if 'x_min' in locals():
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h

            bounding_boxes.append((class_id, x_center, y_center, bbox_width, bbox_height))

    return image, bounding_boxes

def process_single_image(input_path, output_folder):
    """
    Verarbeitet ein einzelnes Bild mit zufälligen Formen, Lupeneffekt,
    Schatten und Rauschen
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

    # Lupeneffekt hinzufügen
    lens_image = add_complex_lens_effect(annotated_image)

    # Schatten hinzufügen
    shadow_image = add_random_shadow(lens_image)

    # Rauschen hinzufügen
    final_image = add_noise(shadow_image)

    # Bild speichern
    cv2.imwrite(output_image_path, final_image)

    # Bounding Boxes speichern
    with open(output_txt_path, "w") as f:
        for box in bounding_boxes:
            f.write(" ".join(map(str, box)) + "\n")

    print(f"Verarbeitet: {filename}")

# Rest des Codes bleibt unverändert
def process_images(input_folder, output_folder, max_workers=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_image, image_path, output_folder)
            for image_path in image_paths
        ]

        concurrent.futures.wait(futures)

if __name__ == "__main__":
    input_folder = "Gen_input_images"
    output_base = "Gen_output_images"

    # Stelle sicher, dass der Hauptordner existiert
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for i in range(2):  # Zwei Iterationen: Train und Validate
        if i == 0:
            output_folder = os.path.join(output_base, "train")
        else:
            output_folder = os.path.join(output_base, "validate")
        print(f"Iteration {i} gestartet: Verarbeite Bilder in {output_folder}")

        process_images(input_folder, output_folder)

