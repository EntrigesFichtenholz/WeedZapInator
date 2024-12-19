import cv2
import numpy as np
import os
import json
import torch
import time
from ultralytics import YOLO
import statistics
from pathlib import Path
import logging
import mmap
from threading import Timer

# Konfiguration
SETTINGS_FILE = "settings.txt"
MODEL_PATH = "./savemodels/MultiRes302ZyklenModel_20241215_024611_20241215_043825.pt"
CALIBRATION_FRAMES = 30
CALIBRATION_TIMEOUT = 3
MIN_CONFIDENCE = 0.5
KNOWN_RECTANGLE_SIZE = 3.0  # cm

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

def load_settings():
    """Lädt die Kalibrierungseinstellungen"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as file:
                settings = json.load(file)
                if validate_settings(settings):
                    return settings
    except Exception as e:
        logging.warning(f"Fehler beim Laden der settings.txt: {str(e)}")
    
    return {"scale_x": 32.10, "scale_y": 30.10}

def save_settings(settings):
    """Speichert die Kalibrierungseinstellungen"""
    if not validate_settings(settings):
        print("Ungültige Kalibrierungswerte - Speicherung abgebrochen")
        return False
        
    try:
        # Backup erstellen
        if os.path.exists(SETTINGS_FILE):
            backup_file = SETTINGS_FILE + ".backup"
            Path(SETTINGS_FILE).rename(backup_file)
        
        # Neue Einstellungen speichern
        with open(SETTINGS_FILE, "w") as file:
            json.dump(settings, file, indent=4)
        
        # Backup löschen
        if os.path.exists(SETTINGS_FILE + ".backup"):
            os.remove(SETTINGS_FILE + ".backup")
            
        print(f"Neue Kalibrierung gespeichert: scale_x={settings['scale_x']:.2f}, scale_y={settings['scale_y']:.2f}")
        return True
    
    except Exception as e:
        logging.error(f"Fehler beim Speichern: {str(e)}")
        if os.path.exists(SETTINGS_FILE + ".backup"):
            Path(SETTINGS_FILE + ".backup").rename(SETTINGS_FILE)
        return False

def validate_settings(settings):
    """Überprüft die Gültigkeit der Einstellungen"""
    if not all(key in settings for key in ['scale_x', 'scale_y']):
        return False
    if not all(isinstance(settings[key], (int, float)) for key in ['scale_x', 'scale_y']):
        return False
    if not (10 <= settings['scale_x'] <= 100 and 10 <= settings['scale_y'] <= 100):
        return False
    return True

def draw_calibration_overlay(frame, detected_rectangles):
    """Zeichnet Hilfslinien und Informationen für die Kalibrierung"""
    overlay = frame.copy()
    
    # Zeichne erkannte Rechtecke
    for rect in detected_rectangles:
        x, y = rect
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Zeichne Verbindungslinien zwischen nächsten Rechtecken
    for i, rect in enumerate(detected_rectangles):
        min_dist = float('inf')
        nearest_point = None

        for j, other_rect in enumerate(detected_rectangles):
            if i != j:
                dist = np.linalg.norm(np.array(rect) - np.array(other_rect))
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = (int(other_rect[0]), int(other_rect[1]))
        
        if nearest_point is not None:
            cv2.line(overlay, (int(rect[0]), int(rect[1])), nearest_point, (255, 0, 0), 1)
    
    # Füge Informationstext hinzu
    cv2.putText(overlay, 
                f"Erkannte Rechtecke: {len(detected_rectangles)}", 
                (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2)
    
    return overlay

def calibrate_system(detector, cap):
    """Führt die Systemkalibrierung durch"""
    print("\nKalibrierung startet in 3 Sekunden...")
    print("Drücken Sie eine beliebige Taste zum Abbrechen")
    
    # Timer für automatischen Timeout
    calibration_active = True
    def timeout_handler():
        nonlocal calibration_active
        calibration_active = False
    
    timer = Timer(CALIBRATION_TIMEOUT, timeout_handler)
    timer.start()
    
    # Warte auf möglichen Abbruch
    if cv2.waitKey(CALIBRATION_TIMEOUT * 1000) & 0xFF != 255:
        timer.cancel()
        print("Kalibrierung abgebrochen")
        return None
    
    if not calibration_active:
        print("Timeout - verwende vorherige Werte")
        return None
    
    print("\nKalibrierung läuft...")
    print("Erfasse Rechtecke...")
    
    # Initialisiere Framebuffer
    framebuffer = OptimizedFramebuffer()
    
    # Sammle Daten
    x_distances = []
    y_distances = []
    
    try:
        for frame_count in range(CALIBRATION_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_resized = cv2.resize(frame, (framebuffer.width, framebuffer.height))
            results = detector(frame_resized, verbose=False)[0]
            
            # Sammle Rechteckpositionen
            rectangles = []
            for box in results.boxes:
                if float(box.conf[0]) > MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    rectangles.append((center_x, center_y))
            
            # Zeichne Overlay
            frame_with_overlay = draw_calibration_overlay(frame_resized, rectangles)
            frame_rgb565 = bgr_to_rgb565(frame_with_overlay)
            framebuffer.write(frame_rgb565.tobytes())
            
            # Berechne Abstände
            if len(rectangles) >= 2:
                # X-Abstände
                rectangles.sort(key=lambda x: x[0])
                for i in range(len(rectangles)-1):
                    x_dist = abs(rectangles[i+1][0] - rectangles[i][0])
                    if 10 < x_dist < 200:  # Filtere unplausible Werte
                        x_distances.append(x_dist)
                
                # Y-Abstände
                rectangles.sort(key=lambda x: x[1])
                for i in range(len(rectangles)-1):
                    y_dist = abs(rectangles[i+1][1] - rectangles[i][1])
                    if 10 < y_dist < 200:  # Filtere unplausible Werte
                        y_distances.append(y_dist)
            
            print(f"Frame {frame_count + 1}/{CALIBRATION_FRAMES}", end='\r')
            
    finally:
        framebuffer.close()
    
    print("\nVerarbeite Messdaten...")
    
    if not x_distances or not y_distances:
        print("Keine ausreichenden Kalibrierdaten gefunden")
        return None
    
    # Berechne neue Skalierungsfaktoren und runde zur Basis 10
    new_scale_x = round(statistics.median(x_distances) / KNOWN_RECTANGLE_SIZE, 1) * 10
    new_scale_y = round(statistics.median(y_distances) / KNOWN_RECTANGLE_SIZE, 1) * 10
    
    # Validiere die neuen Werte
    new_settings = {
        "scale_x": new_scale_x,
        "scale_y": new_scale_y
    }
    
    if not validate_settings(new_settings):
        print("Berechnete Kalibrierungswerte ungültig")
        return None
    
    print(f"\nKalibrierung erfolgreich!")
    print(f"Neue Werte: scale_x = {new_scale_x:.2f}, scale_y = {new_scale_y:.2f}")
    
    return new_settings

def main():
    # Initialisierung
    print("Starte Kalibrierungsprogramm...")
    settings = load_settings()
    print(f"Aktuelle Kalibrierung: scale_x={settings['scale_x']:.2f}, scale_y={settings['scale_y']:.2f}")
    
    try:
        cap = cv2.VideoCapture("/dev/video0")
        if not cap.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden")
            
        detector = YOLO(MODEL_PATH)
        print("Model geladen")
        
        # Führe Kalibrierung durch
        new_settings = calibrate_system(detector, cap)
        
        if new_settings:
            if save_settings(new_settings):
                print("Kalibrierung erfolgreich gespeichert")
            else:
                print("Fehler beim Speichern der Kalibrierung")
        else:
            print("Verwende vorherige Kalibrierung")
            
    except Exception as e:
        print(f"Fehler während der Kalibrierung: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
