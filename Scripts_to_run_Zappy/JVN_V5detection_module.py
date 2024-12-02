import cv2
import numpy as np
import os
import json
import torch
import time
from ultralytics import YOLO
import mmap

SETTINGS_FILE = "settings.txt"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as file:
            return json.load(file)
    return {"scale_x": 36.10, "scale_y": 36.10}  # Default values

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file)

class OptimizedFramebuffer:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.bits_per_pixel = 16
        self.line_length = self.width * 2
        self.size = self.height * self.line_length
        
        self.fb = open('/dev/fb0', 'rb+')
        self.fb_data = mmap.mmap(self.fb.fileno(), self.size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        print(f"Framebuffer initialisiert: {self.width}x{self.height} @ {self.bits_per_pixel}bpp")
    
    def write(self, data):
        self.fb_data.seek(0)
        self.fb_data.write(data)
    
    def close(self):
        self.fb_data.close()
        self.fb.close()

def bgr_to_rgb565(image):
    r = (image[:, :, 2] >> 3).astype(np.uint16) << 11
    g = (image[:, :, 1] >> 2).astype(np.uint16) << 5
    b = (image[:, :, 0] >> 3).astype(np.uint16)
    return (r | g | b).astype(np.uint16)

def draw_ruler_with_scaling_and_framecounter(image, scale_x, scale_y):
    height, width, _ = image.shape
    output_image = image.copy()
    
    for x in range(0, width, int(scale_x)):
        line_length = 20 if x % int(scale_x) == 0 else 10
        cv2.line(output_image, (x, 0), (x, line_length), (255, 255, 255), 2)
        if x % int(scale_x) == 0 and x / scale_x > 0:
            label = f"{x / scale_x:.0f}"
            cv2.putText(output_image, label, (x + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    for y in range(0, height, int(scale_y)):
        line_length = 20 if y % int(scale_y) == 0 else 10
        cv2.line(output_image, (0, y), (line_length, y), (255, 255, 255), 2)
        if y % int(scale_y) == 0 and y / scale_y > 0:
            label = f"{y / scale_y:.0f}"
            cv2.putText(output_image, label, (25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return output_image

def draw_box_with_label_and_scale(image, box, class_name, conf, scale_x, scale_y):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    cm_x = round(center_x / scale_x, 2)
    cm_y = round(center_y / scale_y, 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    label = f"{class_name}: {conf:.2f} (x:{cm_x}cm, y:{cm_y}cm)"
    
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    cv2.rectangle(
        image,
        (x1, y1 - label_height - 10),
        (x1 + label_width, y1),
        (255, 0, 0),
        -1
    )
    
    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )
    
    #print(f"Detected {class_name} at {cm_x}cm x {cm_y}cm with confidence {conf:.2f}")
    
    return image

def calculate_fps(prev_time):
    curr_time = time.time()
    fps = round(float(1 / (curr_time - prev_time)), 5)
    return fps, curr_time

def draw_fps(image, fps):
    cv2.putText(
        image,
        f"FPS: {fps:.4f}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
def laseranalyst():
    settings = load_settings()
    scale_x = settings["scale_x"]
    scale_y = settings["scale_y"]

    cap = cv2.VideoCapture(0)
    detector = YOLO("./savemodels/LightPlantViewer_20241109_224033.pt")
    framebuffer = OptimizedFramebuffer()

    prev_time = time.time()

    # Initialisierung der Liste außerhalb der Schleife
    index_of_found_objekts = []
    start_time = time.time()
    try:
        while time.time() - start_time < 5: #Zeit setzen um die detection vernünftig durchzuführen
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (framebuffer.width, framebuffer.height))
            results = detector(frame_resized, verbose=False)[0]
            frame_with_assets = draw_ruler_with_scaling_and_framecounter(frame_resized, scale_x, scale_y)

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > 0.5:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    # Berechnung der cm-Koordinaten für Druckausgabe
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)   #-5 wegen ofsett
                    center_y = int(((y1 + y2) / 2)) #5Wegen dem Cam offsett (habe ich rausgenommen, da wir das ofsett jetzt im main script definieren)
                    cm_x = round(center_x / scale_x, 2)
                    cm_y = round(center_y / scale_y, 2)

                    # Hinzufügen des Objekts als Tupel oder Dictionary zur Liste
                    index_of_found_objekts.append((class_name, cm_x, cm_y, conf))
                    
                    frame_with_assets = draw_box_with_label_and_scale(
                        frame_with_assets, box, class_name, conf, scale_x, scale_y
                    )

            fps, prev_time = calculate_fps(prev_time)
            draw_fps(frame_with_assets, fps)

            frame_rgb565 = bgr_to_rgb565(frame_with_assets)
            framebuffer.write(frame_rgb565.tobytes())

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Programm beendet.")
                break
            elif key == ord('x'):
                scale_x = float(input("Neue Skalierung für X-Achse (Pixel pro cm): "))
                settings["scale_x"] = scale_x
                save_settings(settings)
                print(f"Neue Skalierung X-Achse: {scale_x}")
            elif key == ord('y'):
                scale_y = float(input("Neue Skalierung für Y-Achse (Pixel pro cm): "))
                settings["scale_y"] = scale_y
                save_settings(settings)
                print(f"Neue Skalierung Y-Achse: {scale_y}")
    finally:
        framebuffer.close()
        cap.release()

    # Rückgabe der Liste nach Schließen der Ressourcen
    return index_of_found_objekts

if __name__ == "__main__":
    laseranalyst()
