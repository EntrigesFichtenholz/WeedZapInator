# WeedZapInator (Zappi) - Autonomes Unkraut-Erkennungs- und Eliminationssystem

## Projektübersicht
Der WeedZapInator ist ein System, das darauf ausgelegt ist, Unkraut autonom zu erkennen und mit Lasertechnologie zu beseitigen. Das Ziel ist eine umweltfreundliche und effiziente Alternative zu herkömmlichen Unkrautbekämpfungsmethoden.

## Benötigte Python-Bibliotheken

### Hauptbibliotheken
- `numpy`: Numerische Berechnungen und Arrayoperationen
- `opencv-python` (cv2): Bildverarbeitung und Computervision
- `torch`: Deep Learning und neuronale Netze
- `ultralytics`: YOLO-Objekterkennung
- `serial`: Serielle Kommunikation mit dem Laser-Plotter
- `sklearn`: Maschinelles Lernen (Clustering)
- `threading`: Parallele Verarbeitung
- `time`: Zeitmanagement
- `queue`: Threadsichere Warteschlangen
- `statistics`: Statistische Berechnungen
- `mmap`: Speicherabbildung für Framebuffer
- `json`: Konfigurationsspeicherung
- `dataclasses`: Datenklassen
- `enum`: Aufzählungstypen

### Installation
```bash
pip install numpy opencv-python torch ultralytics pyserial scikit-learn
```

## Systemarchitektur
Sie Benötigen dringend ein Linux basierendes System wegen dem Frabebuffer! fb0 muss FullHD sein!
### Modulübersicht

1. **Detektionsmodul (`JVN_V6detection_module.py`)**
   - Verwendung von YOLOv8 zur Objekterkennung
   - Fortschrittliche Bildverarbeitungstechniken
   - Kamera-Kalibrierung und Verzerrungskorrektur
   - Präzise Positionsberechnung von Objekten
   - Generierung annotierter Frames

2. **Laser-Plotter-Steuerungsmodul (`JVN_V4LaserTx.py`)**
   - Serielle Kommunikation mit dem Laser-Plotter
   - Verwaltung des Maschinenkoordinatensystems
   - Bewegungsanalyse und Positionsverfolgung
   - Bewegungsgrenzen-Schutz
   - Absoluter und relativer Koordinatenmodus

3. **Mustergenerator-Modul (`JVN_V2patternizer.py`)**
   - Generierung verschiedener geometrischer Bewegungsmuster
   - Unterstützt Kreis-, Spiralen-, Raster- und Zufallsmuster
   - Erstellung von G-Code-Befehlen für präzise Bewegungen

4. **Hauptausführungsmodul (`JVN_V8LaserRunner.py`)**
   - Orchestrierung des gesamten Workflows
   - Threading-Management für Parallelverarbeitung
   - Implementierung von Objekt-Clustering und Median-Berechnung
   - Koordination der Laser-Plotter-Bewegungen

## Detaillierter Workflow

### Erkennungs- und Zielerfassungsprozess
1. Laser-Plotter initialisieren und Verbindung herstellen
2. Initielles Homing der Maschine durchführen
3. Videoframes erfassen und Objekterkennung starten
4. Erkannte Objekte verarbeiten:
   - Koordinatenskalierung anwenden
   - Linsenverzerrungen korrigieren
   - Ähnliche Erkennungen clustern
   - Medianpositionen berechnen
5. Präzise Bewegungsbefehle berechnen
6. Laser-Plotter zu Unkraut-Positionen dirigieren

## Konfiguration

### Kalibrierung
- `settings.txt`: Skalierungskalibrierung
- Anpassbare Parameter in Detektions- und Plotter-Modulen
- Konfigurations-Optionen für Mustergenerierung
   """Das folgende ist ein Audiotranskript, welches ich auch ins Github schreibe:
   
   In diesem Abschnitt erläutere ich, wie der Laserplotter zu kalibrieren ist.
   Der Prozess basiert im Wesentlichen auf zwei Skriptdateien, die zum Einsatz kommen: „laserTX“ in der Version 4.
   In dieser Datei finden Sie am unteren Ende Parameter, mit denen Sie den Raum anpassen können, den der Laser nach dem „Homing“ einnimmt.
   
   Um zu überprüfen, ob alle Einstellungen korrekt sind, können Sie ein kleines Rechteck nahe dem Koordinatenursprung testen. Sobald dies stimmig ist, verschieben Sie bitte Ihre Kalibrierkarte zum dritten Viereck/Rechteck am Ende des Koordinatenursprungs. Nun können Sie den Maßstab für die X- und Y-Achse anpassen.
   
   Dabei ist zu beachten, dass die X-Achse auf dem Bildschirm horizontal und die Y-Achse vertikal verläuft, während es beim Plotter umgekehrt ist. Das Programm berücksichtigt dies und tauscht die Achsen entsprechend um. Ich belasse diese Einstellung derzeit so, da sie das Plotterbild präzise abbildet.
   
   Zur besseren Verständlichkeit werde ich Kommentare im Code hinzufügen. Geplant ist ohnehin, den Code detaillierter zu dokumentieren. Der Kalibrierungsvorgang erfordert mehrfaches Wechseln zwischen der Einstellung des Offsets und des Maßstabs. Diesen Prozess wiederholen Sie bitte dreimal für jede Achse, um die Kalibrierung abzuschließen."""

## Hardwareanforderungen
- USB-Verbindung zum Laser-Plotter
- Kalibrierte Kamera
- CUDA-kompatible GPU (empfohlen)

## Sicherheit und Einschränkungen
- Betrieb innerhalb definierter Maschinengrenzen
- Bewegungsvalidierung implementiert
- Erfordert sorgfältige Kalibrierung für präzise Zielerfassung

## Technische Besonderheiten
- Verwendung von DBSCAN für Clustering
- Dynamische G-Code-Generierung
- Framebuffer-Rendering
- Serielle Kommunikationsprotokoll-Handling

## Herausforderungen und Lösungsansätze
- Präzise Koordinatentransformation
- Echtzeitbildverarbeitung
- Robuste Objekterkennung trotz variabler Umgebungsbedingungen

## Zukünftige Verbesserungen
- Verbesserte Objektklassifizierung
- Dynamische Mustergenerierung
- Erweiterte Clustering-Algorithmen
- Erhöhte Maschinenkompatibilität

## Entwickler
Johannes Nitschke - Projektleiter
