# WeedZapInator (Zappi) - Autonomes Unkraut-Erkennungs- und Eliminationssystem

## Projektübersicht
Der WeedZapInator ist ein innovatives robotisches System, das darauf ausgelegt ist, Unkraut autonom zu erkennen und mit Lasertechnologie zu beseitigen. Das Ziel ist eine umweltfreundliche und effiziente Alternative zu herkömmlichen Unkrautbekämpfungsmethoden.

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
