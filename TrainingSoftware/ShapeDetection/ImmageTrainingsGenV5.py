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

def generate_shapes_with_bounding_boxes_until_valid(image, num_shapes=3, margin=100):
    """
    Generiert Formen, bis mindestens eine Form erfolgreich auf das Bild angewendet wurde.
    Schatten werden ausgeschlossen.
    """
    while True:
        modified_image = image.copy()  # Bildkopie, um wiederholt zu generieren
        annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes(modified_image, num_shapes, margin)

        # Überprüfen, ob valide Shapes generiert wurden
        if bounding_boxes:  # Prüfen, ob mindestens eine Bounding-Box existiert
            return annotated_image, bounding_boxes
        print("Keine gültigen Formen generiert, versuche erneut...")



def generate_shapes_with_bounding_boxes(image, num_shapes=18, margin=100):
    """
    Generiert zufällige Formen (Rechtecke, Kreise, Dreiecke) und gibt die Bounding Boxes im YOLO-Format zurück.
    Schatten werden ausgeschlossen.
    """
    h, w, _ = image.shape
    bounding_boxes = []

    for _ in range(num_shapes):
        # Zufällige Ankerpunkte innerhalb der Margins
        anchor_x = random.randint(margin, w - margin)
        anchor_y = random.randint(margin, h - margin)
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(['circle', 'square', 'triangle'])

        if shape_type == 'circle':
            max_radius = max(10, min(anchor_x - margin, w - anchor_x - margin, anchor_y - margin, h - anchor_y - margin) // 2)
            radius = random.randint(10, max_radius)

            # Kreis zeichnen
            center = (anchor_x, anchor_y)
            cv2.circle(image, center, radius, color, -1)

            # YOLO-Bounding-Box
            x_center = anchor_x / w
            y_center = anchor_y / h
            bbox_width = 2 * radius / w
            bbox_height = 2 * radius / h
            bounding_boxes.append((0, x_center, y_center, bbox_width, bbox_height))

        elif shape_type == 'square':
            max_side = max(10, min(anchor_x - margin, w - anchor_x - margin, anchor_y - margin, h - anchor_y - margin) // 2)
            side = random.randint(10, max_side)
            x1 = anchor_x - side // 2
            y1 = anchor_y - side // 2
            x2 = anchor_x + side // 2
            y2 = anchor_y + side // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

            # YOLO-Bounding-Box
            x_center = anchor_x / w
            y_center = anchor_y / h
            bbox_width = side / w
            bbox_height = side / h
            bounding_boxes.append((1, x_center, y_center, bbox_width, bbox_height))

        elif shape_type == 'triangle':
            max_size = min(anchor_x - margin, w - anchor_x - margin, anchor_y - margin, h - anchor_y - margin) // 2
            pt1 = (anchor_x, anchor_y - max_size // 2)
            pt2 = (anchor_x - max_size // 2, anchor_y + max_size // 2)
            pt3 = (anchor_x + max_size // 2, anchor_y + max_size // 2)
            cv2.fillPoly(image, [np.array([pt1, pt2, pt3], dtype=np.int32)], color)

            # YOLO-Bounding-Box
            x_min = min(pt1[0], pt2[0], pt3[0]) / w
            x_max = max(pt1[0], pt2[0], pt3[0]) / w
            y_min = min(pt1[1], pt2[1], pt3[1]) / h
            y_max = max(pt1[1], pt2[1], pt3[1]) / h
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bounding_boxes.append((2, x_center, y_center, bbox_width, bbox_height))

    return image, bounding_boxes


def process_images(input_folder, output_folder):
    """
    Verarbeitet alle Bilder im Eingabeordner und speichert die Ausgabebilder sowie YOLO-Bounding-Boxes.
    Schatten werden ausgeschlossen.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"Fehler beim Laden des Bildes: {filename}")
            continue

        # Zufällige Formen und Bounding Boxes (wiederholen bis gültig)
        annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes_until_valid(image)

        # Effekte hinzufügen
        shadow_image = add_random_shadow(annotated_image)  # Schatten nach den Formen hinzufügen
        lens_image = add_complex_lens_effect(shadow_image)
        final_image = add_noise(lens_image)

        # Speichern der Bilder
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, final_image)

        # Speichern der Bounding Boxes (ohne Schatten)
        output_txt_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
        with open(output_txt_path, "w") as f:
            for box in bounding_boxes:
                f.write(" ".join(map(str, box)) + "\n")

        print(f"Verarbeitet: {filename}")



if __name__ == "__main__":
    input_folder = "Gen_input_images"
    output_base = "Gen_output_images"

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for i in range(2):  # Zwei Iterationen: Train und Validate
        output_folder = os.path.join(output_base, "train" if i == 0 else "validate")
        process_images(input_folder, output_folder)
