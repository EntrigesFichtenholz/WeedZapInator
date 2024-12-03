import threading
import queue
import statistics
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from JVN_V4LaserTx import lasercommander, LaserPlotterTx
from JVN_V6detection_module import laseranalyst
from JVN_V2patternizer import CirclePatternizer, SpiralPatternizer, GridPatternizer, RandomPatternizer

#Niedriger eps ist ungenauer
def cluster_and_median_tuples(found_objects, eps=0.05, min_samples=2):
    # Wenn found_objects eine flache Liste ist, in Liste umwandeln
    if found_objects and all(isinstance(obj, tuple) for obj in found_objects):
        found_objects = [found_objects]
    
    # Alle Tupel in eine einzelne Liste umwandeln
    all_tuples = [item for sublist in found_objects for item in sublist]
    
    # Extrahiere Koordinaten für Clustering
    coords = np.array([(t[1], t[2]) for t in all_tuples])
    
    # Skaliere die Koordinaten für besseres Clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # DBSCAN Clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    
    # Gruppiere Tupel basierend auf Clustering
    clustered_tuples = {}
    for i, label in enumerate(clustering.labels_):
        if label != -1:  # Ignoriere Noise-Punkte
            if label not in clustered_tuples:
                clustered_tuples[label] = []
            clustered_tuples[label].append(all_tuples[i])
    
    # Berechne Median für jede Cluster
    median_results = []
    for cluster, cluster_tuples in clustered_tuples.items():
        # Extrahieren der einzelnen Komponenten für diesen Cluster
        classes = [t[0] for t in cluster_tuples]
        xs = [t[1] for t in cluster_tuples]
        ys = [t[2] for t in cluster_tuples]
        confidences = [t[3] for t in cluster_tuples]
        
        # Berechnung der Mediane für numerische Werte
        median_x = statistics.median(xs)
        median_y = statistics.median(ys)
        median_confidence = statistics.median(confidences)
        
        # Ermittlung des häufigsten Klassennamens
        most_common_class = max(set(classes), key=classes.count)
        
        # Erstellen des Median-Tupels für diesen Cluster
        median_tuple = (most_common_class, median_x, median_y, median_confidence)
        median_results.append(median_tuple)
    
    return median_results, clustering.labels_

def plotter_thread(plotter, command_queue, position_queue, stop_event):
    """Thread to manage the LaserPlotter commands."""
    try:
        while not stop_event.is_set():
            try:
                # Wait for a command from the queue, with a timeout
                command = command_queue.get(timeout=0.1)
                if command == "exit":
                    break
                elif command == "home":
                    plotter.home()
                elif command == "pos":
                    position_and_status = plotter.get_position_and_status()
                    #Status is refering to coordinate_mode
                    x, y, status = position_and_status
                    position_queue.put((x,y))
                else:
                    response = plotter.send_gcode(command)
                    print(f"Laserplotter antwortete: {response}")
            except queue.Empty:
                # No command received, continue checking
                continue
    finally:
        plotter.close_connection()
        print("Plotter-Thread wurde beendet.")

def main():
    #Einstellungen für den Patternizer nach Ansteuerung auf Objekt # Generiert Muster wenn ueber Objekt. (Fuktioniert aber noch nicht so richtig)
    #Deaktiviert, da es nicht so richtig mit mehreren Objekten fuktioniert :/
    pattenizer_activate = False
    circle_pattern = CirclePatternizer(
    radius=0.5,
    radius_step=0.5,   # Schrittweite zwischen Kreisen
    steps=12)
    
    patternizer_commands = circle_pattern.generate_gcode()
    
    #Alternatives pattern noch nicht konfiguriert !!!!
    """
    pattern = SpiralPatternizer(
    max_radius=50,     # Maximaler Radius der Spirale in mm
    radius_step=2,     # Schrittweise Vergrößerung des Radius pro Umdrehung
    steps=36           # Anzahl der Schritte pro Kreisumrundung
    )
    """
    print("Starte Laser Commander...")
    plotter = lasercommander()
    if plotter is None:
        print("Verbindung zum Laserplotter konnte nicht hergestellt werden.")
        return

    # Create a thread-safe queue for commands
    command_queue = queue.Queue()

    # Event to signal the thread to stop
    stop_event = threading.Event()

    # Create a queue for results
    position_queue = queue.Queue()

    # Start the plotter thread
    thread = threading.Thread(target=plotter_thread, args=(plotter, command_queue, position_queue, stop_event))
    thread.start()
    time.sleep(10)
    
    #Durch tauschen der Aderpare am Y Steppermotor nicht notwendig :)  Um 300 mm nachoben verschieben, damit das feld nicht verdeckt ist
    #command_queue.put(str("G91 Y290 F0"))  # Send command to the thread
    
    # Warte auf das Ergebnis
    #x, y = result_queue.get()
    command_queue.put(str("pos"))
    x, y = position_queue.get()
    print(f"XYPOS: X={x}, Y={y}")
    # Hole Detected Objects nach 5 sekunden wegen des homings
    time.sleep(3)
    
    #Objekte erkennen, nach max versuchen wird abgebrochen!
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
          found_objects = laseranalyst()
          attempts += 1
          if found_objects and all(isinstance(item, tuple) for item in found_objects):
             break  # Erfolgreiche Ergebnisse gefunden, Schleife abbrechen
          print(f"Attempt {attempts}: No valid objects found, retrying...")

    if not found_objects or not all(isinstance(item, tuple) for item in found_objects):
        print("Max attempts reached. No valid objects found.")
    else:
        print("Objekte erkannt!")
    
    try:
        # Clustering und Median-Berechnung
        # Passen Sie eps und min_samples nach Bedarf an
        # - eps: maximale Distanz zwischen zwei Punkten in einem Cluster
        # - min_samples: minimale Anzahl von Punkten, um einen Cluster zu bilden
        median_tuples, labels = cluster_and_median_tuples(found_objects, eps=0.5, min_samples=2)
        
        # Ausgabe der geclusterten Mediantupel
        print("Clusterergebnisse:")
        for i, median_tuple in enumerate(median_tuples, 1):
            print(f"Cluster {i}: {median_tuple}")
            object_name = median_tuple[0]
            x_coord_object = median_tuple[1]
            y_coord_object = median_tuple[2]
            conf_object = median_tuple[3]
            
            #Holen der aktuellen Position
            command_queue.put(str("pos"))
            current_x, current_y = position_queue.get()

            #Current pos in mm aber object pos in cm
            new_x_coord_command = round((x_coord_object*10 - current_x-80), 2) #Laserofsett -80
            new_y_coord_command = round((y_coord_object*10 - current_y), 2)
            print(f"x_coord_object{x_coord_object*10}-current_x{current_x}")
            try:
               command = str(f"G91 X{new_x_coord_command} Y{new_y_coord_command} F0")
               command_queue.put(command)  # Send command to the thread
               #logik falls ein menschlicher finger erkannt werden sollte(NOCH NICHT IMPLEMENTIERT)
               if command.lower() == "exit":
                  stop_event.set()  # Signal the thread to exit
                  break

            except KeyboardInterrupt:
                print("Programm wird beendet...")
                stop_event.set()
            time.sleep(0.5)
            
            if pattenizer_activate is True:
               # Sende die G-Code-Befehle an den Laserplotter 
               for patternizer_command in patternizer_commands:
                   command_queue.put(patternizer_command)  # Füge jeden Befehl einzeln hinzu
                   time.sleep(0.01)  # Optionale Pause zwischen den Befehlen
               time.sleep(1)
            #command_queue.put('pos')  # Send command to the thread
        """
        # Optional: Detaillierte Informationen zu den Clustern
        if labels is not None:
            all_tuples = [item for sublist in found_objects for item in sublist]
            for label in set(labels):
                if label != -1:
                    cluster_points = [all_tuples[j] for j, l in enumerate(labels) if l == label]
                    print(f"\nCluster {label} Punkte:")
                    for point in cluster_points:
                        print(point)
        """

    except Exception as e:
        print(f"Fehler in der main funktion :( : {e}")
        return
    
    """    
    try:
        while True:
            command = input("Geben Sie einen G-Code-Befehl ein (oder 'exit' zum Beenden): ").strip()
            command_queue.put(command)  # Send command to the thread
            
            if command.lower() == "exit":
                stop_event.set()  # Signal the thread to exit
                break
    
    except KeyboardInterrupt:
        print("Programm wird beendet...")
        stop_event.set()
    
    finally:
        thread.join()  # Wait for the thread to finish
        print("Programm beendet.")
    """
if __name__ == "__main__":
    main()
