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

def add_shadow_gradient(image):
    """
    Fügt einen Schattenverlauf zum Bild hinzu, um unterschiedliche Belichtungen zu simulieren.
    """
    h, w, _ = image.shape

    # Zufällige Parameter für den Verlauf
    gradient_type = random.choice(['horizontal', 'vertical', 'diagonal'])
    start_intensity = random.uniform(0.3, 0.8)  # Starttransparenz des Schattens
    end_intensity = random.uniform(0.0, 0.5)    # Endtransparenz des Schattens

    # Erstelle eine Schattenmaske
    shadow_mask = np.zeros((h, w), dtype=np.float32)

    if gradient_type == 'horizontal':
        for i in range(h):
            alpha = start_intensity + (end_intensity - start_intensity) * (i / h)
            shadow_mask[i, :] = alpha

    elif gradient_type == 'vertical':
        for i in range(w):
            alpha = start_intensity + (end_intensity - start_intensity) * (i / w)
            shadow_mask[:, i] = alpha

    elif gradient_type == 'diagonal':
        for i in range(h):
            for j in range(w):
                alpha = start_intensity + (end_intensity - start_intensity) * ((i + j) / (h + w))
                shadow_mask[i, j] = alpha

    # Konvertiere Maske in 3 Kanäle
    shadow_mask = cv2.merge([shadow_mask] * 3)

    # Wende die Maske auf das Bild an
    shadowed_image = cv2.convertScaleAbs(image * (1 - shadow_mask))
    return shadowed_image

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
    strength = np.random.uniform(0.000005, 0.000099)
    center_offset_x = random.randint(-w // 8, w // 8)
    center_offset_y = random.randint(-h // 8, h // 8)
    scale_factor = random.uniform(0.9, 1.05)
    """
    print(f"Effektparameter: Stärke={strength:.6f}, "
          f"Zentrum-Offset=({center_offset_x}, {center_offset_y}), "
          f"Skalierungsfaktor={scale_factor:.2f}")"""

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

def add_chromatic_aberration(image):
    """
    Fügt chromatische Aberration hinzu, indem RGB-Kanäle verschoben werden.
    """
    h, w, _ = image.shape
    shift = random.randint(1, 5)  # Zufällige Verschiebung
    b, g, r = cv2.split(image)

    b = np.roll(b, shift, axis=1)  # Blaukanal verschieben
    r = np.roll(r, -shift, axis=1)  # Rotkanal verschieben

    aberrated_image = cv2.merge((b, g, r))
    return aberrated_image

def adjust_gamma(image, gamma=None):
    """
    Passt den Gamma-Wert des Bildes an.
    Wenn kein Gamma-Wert übergeben wird, wird ein zufälliger Wert zwischen 0.5 und 1.5 verwendet.
    """
    if gamma is None:
        gamma = random.uniform(0.5, 1.5)  # Zufälliger Gamma-Wert
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table), gamma

def apply_motion_blur(image, kernel_size=None):
    """
    Wendet Bewegungsunschärfe auf ein Bild an.
    """
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7, 9])  # Zufällige Kernelgröße

    # Erstelle den Bewegungsunschärfe-Kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Wende den Kernel auf das Bild an
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def add_jpeg_compression(image):
    """
    Simuliert JPEG-Komprimierungsartefakte.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 70)]  # Zufällige Qualitätsstufe
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encoded_image, 1)
    return compressed_image

def apply_sepia_filter(image):
    """
    Wendet einen Sepia-Filter mit zufälligen Kernelwerten auf das Bild an.
    """
    # Generiere zufällige Kernelwerte zwischen 0 und 1 und ja, 3 und 3 ist richtig so.. trust me ^^ Johannes :)
    kernel = np.random.rand(3, 3)
    # Normalisiere den Kernel, sodass die Summe nahe 1 liegt
    kernel = kernel / kernel.sum()

    sepia_image = cv2.transform(image, kernel)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def apply_color_tint(image, tint_type=None):
    """
    Fügt dem Bild einen sanften Farbton hinzu.
    """
    b, g, r = cv2.split(image)

    # Wenn kein tint_type angegeben, wähle zufällig
    if tint_type is None:
        tint_type = np.random.choice(['warm', 'cool', 'random'])

    if tint_type == 'warm':
        # Sehr subtile Verstärkung für rot und Reduzierung für blau
        r_adjust = np.random.randint(5, 20)
        b_adjust = np.random.randint(1, 10)
        r = cv2.add(r, r_adjust)
        b = cv2.subtract(b, b_adjust)
    elif tint_type == 'cool':
        # Sehr subtile Verstärkung für blau und Reduzierung für rot
        r_adjust = np.random.randint(1, 10)
        b_adjust = np.random.randint(5, 20)
        r = cv2.subtract(r, r_adjust)
        b = cv2.add(b, b_adjust)
    elif tint_type == 'random':
        # Sehr subtile zufällige Farbverschiebungen
        r_adjust = np.random.randint(-15, 15)
        g_adjust = np.random.randint(-15, 15)
        b_adjust = np.random.randint(-15, 15)

        r = cv2.add(r, r_adjust)
        g = cv2.add(g, g_adjust)
        b = cv2.add(b, b_adjust)

    tinted_image = cv2.merge((b, g, r))
    return tinted_image

def process_image(input_path, output_folder):
    """
    Verarbeitet ein einzelnes Bild.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {input_path}")
        return

    annotated_image, bounding_boxes = generate_shapes_with_bounding_boxes(image)

    # Schattenverlauf hinzufügen
    shadowed_image = add_shadow_gradient(annotated_image)

    distorted_image, updated_bounding_boxes = add_lens_effect(shadowed_image, bounding_boxes)

    noisy_image = add_noise(distorted_image)
    motion_blurred_image = apply_motion_blur(noisy_image)

    choice = random.randint(1, 4)  # Wählt eine Zahl zwischen 1 und 3 aus

    if choice == 1:
        tinted_image = apply_color_tint(motion_blurred_image, tint_type='cool')
        sepia_image = apply_sepia_filter(tinted_image)
    else:
        sepia_image = motion_blurred_image

    compressed_image = add_jpeg_compression(sepia_image)
    aberrated_image = add_chromatic_aberration(compressed_image)

    filename = os.path.basename(input_path)
    output_image_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_image_path, compressed_image)

    output_txt_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
    with open(output_txt_path, "w") as f:
        for box in updated_bounding_boxes:
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

def process_images_multithreaded(input_folder, output_folder, num_threads=12):
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


#Oh.. das ist doch länger geworden als gedacht... und man braucht eine sehr starke Cpu, ich hätte es auf die Graka auslegen sollen...
