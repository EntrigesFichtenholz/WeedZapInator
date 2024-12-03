import numpy as np
#Kalibriert den verstz der durch die optik der Kamera mittels ihrer Linse hervorgerufen wird.
#Ermittelte werte müssen per hand in detection.py eingetragen werden unnter der Variable CAMERA_MATRIX
# Angenommene Werte
sensor_size_x = 5  # Sensorgröße in mm (horizontal)
sensor_size_y = 3.75  # Sensorgröße in mm (vertikal)
resolution_x = 1920  # Pixel (horizontal)
resolution_y = 1080  # Pixel (vertikal)

# Separate Skalierungsfaktoren für X und Y
x_scaling_factor = 0.8333  # Skalierungsfaktor (wie 3 cm auf 2,5 cm abgebildet) also auf dem Bildschirm
y_scaling_factor = 0.6667  # Skalierungsfaktor (wie 3 cm auf 2,5 cm abgebildet) also auf dem Bildschirm

# Berechnung der Fokallängen in Pixeln
f_x = (sensor_size_x * resolution_x) / (sensor_size_x * x_scaling_factor)  
f_y = (sensor_size_y * resolution_y) / (sensor_size_y * y_scaling_factor)

# Berechnung des optischen Mittelpunkts
c_x = resolution_x / 2
c_y = resolution_y / 2

# Kamera-Matrix
CAMERA_MATRIX = np.array([[f_x, 0, c_x], 
                          [0, f_y, c_y], 
                          [0, 0, 1]], dtype=np.float32)

print(CAMERA_MATRIX)