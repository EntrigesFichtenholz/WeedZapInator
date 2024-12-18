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
from math import sqrt

def cluster_objects(found_objects, eps=0.5, min_samples=2):
    """
    Führt Clustering auf den Objekten durch, basierend auf ihren 2D-Koordinaten.
    :param found_objects: Liste von Tupeln (class_name, cm_x, cm_y, confidence).
    :param eps: Maximaler Abstand zwischen zwei Punkten, um sie in einem Cluster zu gruppieren.
    :param min_samples: Minimale Anzahl von Punkten, um ein Cluster zu bilden.
    :return: Liste von Mediantupeln (class_name, cm_x, cm_y, confidence) und Cluster-Labels.
    """
    if not found_objects:
        return [], []

    coordinates = np.array([[obj[1], obj[2]] for obj in found_objects])  # cm_x, cm_y

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

    clustered_tuples = {}
    for i, label in enumerate(clustering.labels_):
        if label != -1:  # Ignoriere Noise (-1)
            if label not in clustered_tuples:
                clustered_tuples[label] = []
            clustered_tuples[label].append(found_objects[i])

    median_results = []
    for cluster, objects in clustered_tuples.items():
        classes = [obj[0] for obj in objects]
        xs = [obj[1] for obj in objects]
        ys = [obj[2] for obj in objects]
        confidences = [obj[3] for obj in objects]

        median_x = statistics.median(xs)
        median_y = statistics.median(ys)
        median_confidence = statistics.median(confidences)
        most_common_class = max(set(classes), key=classes.count)

        median_results.append((most_common_class, median_x, median_y, median_confidence))

    return median_results, clustering.labels_

def find_optimal_path(coordinates):
    """
    Findet die effizienteste Reihenfolge der Punkte basierend auf der Distanz.
    :param coordinates: Liste von Koordinaten [(x1, y1), (x2, y2), ...].
    :return: Sortierte Liste der Koordinaten [(x1, y1), (x2, y2), ...].
    """
    if not coordinates:
        return []

    def distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    sorted_path = []
    remaining_points = coordinates[:]
    current_point = (0, 0)  # Startpunkt (Nullpunkt)

    while remaining_points:
        next_point = min(remaining_points, key=lambda p: distance(current_point, p))
        sorted_path.append(next_point)
        remaining_points.remove(next_point)
        current_point = next_point

    return sorted_path

def plotter_thread(plotter, command_queue, position_queue, stop_event):
    """Thread to manage the LaserPlotter commands."""
    try:
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.1)
                if command == "exit":
                    break
                elif command == "home":
                    plotter.home()
                elif command == "pos":
                    position_and_status = plotter.get_position_and_status()
                    x, y, status = position_and_status
                    position_queue.put((x, y))
                else:
                    response = plotter.send_gcode(command)
                    print(f"Laserplotter antwortete: {response}")
            except queue.Empty:
                continue
    finally:
        plotter.close_connection()
        print("Plotter-Thread wurde beendet.")

def main():
    enable_clustering = input("Clustering aktivieren? (y/n): ").strip().lower() == 'y'

    print("Starte Laser Commander...")
    plotter = lasercommander()
    if plotter is None:
        print("Verbindung zum Laserplotter konnte nicht hergestellt werden.")
        return

    command_queue = queue.Queue()
    stop_event = threading.Event()
    position_queue = queue.Queue()

    thread = threading.Thread(target=plotter_thread, args=(plotter, command_queue, position_queue, stop_event))
    thread.start()
    time.sleep(10)

    command_queue.put("pos")
    x, y = position_queue.get()
    print(f"XYPOS: X={x}, Y={y}")
    time.sleep(3)

    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        found_objects = laseranalyst()
        attempts += 1
        if found_objects and all(isinstance(item, tuple) for item in found_objects):
            break
        print(f"Attempt {attempts}: No valid objects found, retrying...")

    if not found_objects or not all(isinstance(item, tuple) for item in found_objects):
        print("Max attempts reached. No valid objects found.")
        return
    else:
        print("Objekte erkannt!")

    try:
        if enable_clustering:
            median_tuples, labels = cluster_objects(found_objects, eps=0.5, min_samples=2)
            objects_to_process = median_tuples
        else:
            objects_to_process = found_objects

        coordinates = [(obj[1] * 10, obj[2] * 10) for obj in objects_to_process]
        optimal_path = find_optimal_path(coordinates)

        print("Optimierter Pfad:")
        for i, (x_mm, y_mm) in enumerate(optimal_path, 1):
            print(f"Punkt {i}: X={x_mm}, Y={y_mm}")

            command_queue.put("pos")
            current_x, current_y = position_queue.get()

            new_x = round((x_mm - current_x - 80), 2)
            new_y = round((y_mm - current_y), 2)
            print(f"Command: X={new_x}, Y={new_y}")

            try:
                command_queue.put(f"G91 X{new_x} Y{new_y} F0")
            except KeyboardInterrupt:
                print("Programm wird beendet...")
                stop_event.set()
            time.sleep(0.5)

    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    main()
