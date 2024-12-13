import os
import cv2
import random
import numpy as np
from queue import Queue
from threading import Thread

def generate_shapes_with_bounding_boxes(image, num_shapes=4, margin=20):
    """
    Generiert zufällige Formen (Rechtecke, Kreise, Dreiecke) und gibt die Bounding Boxes im YOLO-Format zurück.
    Formen sind vollständig innerhalb der Bildgrenzen.
    """
    h, w, _ = image.shape
    bounding_boxes = []

    for _ in range(num_shapes):
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(['circle', 'square', 'triangle'])
        anchor_x = random.randint(margin, w - margin)
        anchor_y = random.randint(margin, h - margin)

        if shape_type == 'circle':
            max_radius = min(anchor_x - margin, w - anchor_x - margin,
                             anchor_y - margin, h - anchor_y - margin)
            radius = random.randint(10, max(10, max_radius))
            center = (anchor_x, anchor_y)
            cv2.circle(image, center, radius, color, -1)

            # YOLO-Bounding-Box
            x_center = anchor_x / w
            y_center = anchor_y / h
            bbox_width = 2 * radius / w
            bbox_height = 2 * radius / h
            bounding_boxes.append((0, x_center, y_center, bbox_width, bbox_height))

        elif shape_type == 'square':
            max_side = min(anchor_x - margin, w - anchor_x - margin,
                           anchor_y - margin, h - anchor_y - margin)
            side = random.randint(10, max(10, max_side))
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
            max_size = min(anchor_x - margin, w - anchor_x - margin,
                           anchor_y - margin, h - anchor_y - margin)
            size = random.randint(10, max(10, max_size))
            pt1 = (anchor_x, anchor_y - size // 2)
            pt2 = (anchor_x - size // 2, anchor_y + size // 2)
            pt3 = (anchor_x + size // 2, anchor_y + size // 2)
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

def apply_lens_distortion(image, strength, center_offset_x, center_offset_y, scale_factor):
    """
    Wendet eine Linsenverzerrung auf das Bild an.
    Gibt die verzerrten Koordinatenkarten und das verzerrte Bild zurück.
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # Zentrum der Verzerrung mit Offset
    center_x = w // 2 + center_offset_x
    center_y = h // 2 + center_offset_y

    # Berechnung des Linseneffekts
    scale = np.sqrt((map_x - center_x) ** 2 + (map_y - center_y) ** 2)
    effect = strength * scale * scale_factor

    # Verzerrte Pixelkoordinaten berechnen
    map_x = map_x + (map_x - center_x) * effect
    map_y = map_y + (map_y - center_y) * effect

    # Anwenden der Verzerrung auf das Bild
    distorted_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return distorted_image, map_x, map_y


def adjust_bounding_boxes(bounding_boxes, map_x, map_y, image_shape):
    """
    Passt Bounding-Boxen basierend auf den Verzerrungs-Karten an.
    """
    h, w = image_shape[:2]
    new_bounding_boxes = []

    for cls, x_center, y_center, bbox_width, bbox_height in bounding_boxes:
        # Originale Pixelkoordinaten der Bounding-Box berechnen
        x_min = (x_center - bbox_width / 2) * w
        x_max = (x_center + bbox_width / 2) * w
        y_min = (y_center - bbox_height / 2) * h
        y_max = (y_center + bbox_height / 2) * h

        # Verzerrte Pixelkoordinaten berechnen
        x_min_distorted = map_x[int(y_min), int(x_min)]
        x_max_distorted = map_x[int(y_max), int(x_max)]
        y_min_distorted = map_y[int(y_min), int(x_min)]
        y_max_distorted = map_y[int(y_max), int(x_max)]

        # Neue Bounding-Box im YOLO-Format
        new_x_center = (x_min_distorted + x_max_distorted) / 2 / w
        new_y_center = (y_min_distorted + y_max_distorted) / 2 / h
        new_bbox_width = (x_max_distorted - x_min_distorted) / w
        new_bbox_height = (y_max_distorted - y_min_distorted) / h
        new_bounding_boxes.append((cls, new_x_center, new_y_center, new_bbox_width, new_bbox_height))

    return new_bounding_boxes


def add_lens_effect(image, bounding_boxes=None):
    """
    Fügt einen Linseneffekt hinzu und passt optional die Bounding-Boxen an.
    Alle Effektparameter werden in dieser Funktion berechnet und leicht randomisiert.
    """
    h, w = image.shape[:2]

    # Generiere Effektparameter
    strength = np.random.uniform(0.000005, 0.000099)  # mehr durch np Nachkommastellen
    center_offset_x = random.randint(-w // 8, w // 8)  # Leichter Versatz des Zentrums in x-Richtung
    center_offset_y = random.randint(-h // 8, h // 8)  # Leichter Versatz des Zentrums in y-Richtung
    scale_factor = random.uniform(0.9, 1.05)  # Leichter Skalierungsfaktor

    print(f"Effektparameter: Stärke={strength:.6f}, "
          f"Zentrum-Offset=({center_offset_x}, {center_offset_y}), "
          f"Skalierungsfaktor={scale_factor:.2f}")

    # Verzerrung anwenden
    distorted_image, map_x, map_y = apply_lens_distortion(
        image, strength, center_offset_x, center_offset_y, scale_factor
    )

    if bounding_boxes is not None:
        # Bounding-Boxen anpassen
        updated_bounding_boxes = adjust_bounding_boxes(bounding_boxes, map_x, map_y, image.shape)
        return distorted_image, updated_bounding_boxes

    return distorted_image


def add_noise(image):
    """
    Fügt Rauschen zum Bild hinzu.
    """
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def process_image(input_path, output_folder):
    """
    Verarbeitet ein einzelnes Bild.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {input_path}")
        return

    annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes(image)
    distorted_image, updated_bounding_boxes = add_lens_effect(annotated_image, bounding_boxes)

    noisy_image = add_noise(distorted_image)

    filename = os.path.basename(input_path)
    output_image_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_image_path, noisy_image)

    output_txt_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
    with open(output_txt_path, "w") as f:
        for box in bounding_boxes:
            f.write(" ".join(map(str, box)) + "\n")

    print(f"Verarbeitet: {filename}")

def worker(input_queue, output_folder):
    """
    Arbeiterfunktion für Multithreading.
    """
    while not input_queue.empty():
        input_path = input_queue.get()
        process_image(input_path, output_folder)
        input_queue.task_done()

def process_images_multithreaded(input_folder, output_folder, num_threads=4):
    """
    Verarbeitet Bilder mit Multithreading.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_queue = Queue()
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_queue.put(os.path.join(input_folder, filename))

    threads = []
    for _ in range(num_threads):
        thread = Thread(target=worker, args=(input_queue, output_folder))
        thread.start()
        threads.append(thread)

    input_queue.join()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    input_folder = "Gen_input_images"
    output_base = "Gen_output_images"

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for i in range(2):  # Zwei Iterationen: Train und Validate
        output_folder = os.path.join(output_base, "train" if i == 0 else "validate")
        process_images_multithreaded(input_folder, output_folder)
