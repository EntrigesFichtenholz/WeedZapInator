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

def generate_shapes_with_bounding_boxes(image, num_shapes=12):
    """
    Generiert zufällige Rechtecke basierend auf drei zufälligen Punkten im Zentrum des Bildes,
    mit einem Margin von 100 Pixel vom Rand entfernt. Füllt sie mit einem zufälligen Shape.
    Gibt die Bounding-Box-Koordinaten zurück.
    """
    h, w, _ = image.shape
    center_x, center_y = w / 2, h / 2
    margin = 500
    bounding_boxes = []

    for _ in range(num_shapes):
        # Zufällige Punkte im Zentrum auswählen (innerhalb des Margins vom Rand entfernt)
        points = []
        while len(points) < 3:
            x = random.randint(margin, center_x - margin)
            y = random.randint(margin, center_y - margin)
            if all(abs(x - p[0]) > 100 and abs(y - p[1]) > 100 for p in points):
                points.append((x, y))

        # Rechteck berechnen
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)

        # Zufällige Größe bestimmen
        size_factor = random.uniform(0.05, 1)
        width = int((x_max - x_min) * size_factor)
        height = int((y_max - y_min) * size_factor)

        # Zufällige Position innerhalb des Rechtecks
        x_offset = random.randint(0, x_max - width)
        y_offset = random.randint(0, y_max - height)

        # Zufällige Farbe auswählen
        color = tuple(random.randint(0, 255) for _ in range(3))

        # Rechteck zeichnen
        cv2.rectangle(image, (x_offset, y_offset), (x_offset+width, y_offset+height), color, thickness=-1)

        # Zufälligen Shape generieren
        shape_type = random.choice(['circle', 'square', 'triangle'])

        if shape_type == 'circle':
            center = (x_offset + width//2, y_offset + height//2)
            radius = min(width, height)//2
            cv2.circle(image, center, radius, color, thickness=-1)
        elif shape_type == 'square':
            square_size = min(width, height)//2
            cv2.rectangle(image, (x_offset, y_offset), (x_offset+square_size, y_offset+square_size), color, thickness=-1)
        elif shape_type == 'triangle':
            pt1 = (x_offset, y_offset + height)
            pt2 = (x_offset + width, y_offset + height)
            pt3 = (x_offset + width, y_offset + height)
            cv2.fillPoly(image, [np.array([pt1, pt2, pt3], dtype=np.int32)], color)

        # Bounding Box berechnen
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h

        bounding_boxes.append((color, x_center, y_center, bbox_width, bbox_height))

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

