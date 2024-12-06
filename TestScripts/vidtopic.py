import cv2
import os

# Eingabe- und Ausgabeordner
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_frames"
FRAME_INTERVAL = 15  # Nur jedes 15. Frame speichern

# Ausgabeordner erstellen, falls nicht vorhanden
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alle MOV-Dateien im Eingabeordner durchlaufen
for video_file in os.listdir(INPUT_DIR):
    if video_file.lower().endswith(".mov"):
        video_path = os.path.join(INPUT_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Unterordner für das aktuelle Video erstellen
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Video öffnen
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Fehler: Konnte {video_file} nicht öffnen.")
            continue

        print(f"Verarbeite: {video_file}")
        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Jedes 15. Frame speichern
            if frame_count % FRAME_INTERVAL == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"Frames erfolgreich extrahiert für: {video_file}")

print("Verarbeitung abgeschlossen. Die Bilder befinden sich in:", OUTPUT_DIR)
