import os
import cv2
import random
import numpy as np


def generate_shapes_with_bounding_boxes(image, num_shapes=8, margin=20):
    """
    Generiert zufällige Formen (Rechtecke, Kreise, Dreiecke) und gibt die Bounding Boxes im YOLO-Format zurück.
    Schatten werden ausgeschlossen. Alle Formen sind vollständig innerhalb der Bildgrenzen.
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


def add_random_shadow(image):
    """
    Fügt einen zufälligen Schatten zum Bild hinzu.
    """
    h, w = image.shape[:2]
    alpha = random.uniform(0.1, 0.4)
    shadow_image = image.copy()
    overlay = shadow_image.copy()

    if random.choice(['polygon', 'rectangle']) == 'polygon':
        points = [(random.randint(0, w), random.randint(0, h)) for _ in range(random.randint(3, 6))]
        cv2.fillPoly(overlay, [np.array(points)], (0, 0, 0))
    else:
        x1, y1, x2, y2 = [random.randint(0, dim) for dim in (w, h, w, h)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.addWeighted(overlay, alpha, shadow_image, 1 - alpha, 0, shadow_image)
    return shadow_image


def add_noise(image):
    """
    Fügt Rauschen zum Bild hinzu.
    """
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


def process_images(input_folder, output_folder):
    """
    Verarbeitet alle Bilder im Eingabeordner und speichert die Ausgabebilder sowie YOLO-Bounding-Boxes.
    Schatten werden ausgeschlossen.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Fehler beim Laden des Bildes: {filename}")
            continue

        annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes(image)
        shadow_image = add_random_shadow(annotated_image)
        noisy_image = add_noise(shadow_image)

        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, noisy_image)

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
