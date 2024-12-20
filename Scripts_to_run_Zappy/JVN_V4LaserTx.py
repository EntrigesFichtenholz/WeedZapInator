import serial
import time
from dataclasses import dataclass
from typing import Optional, List

# Maschinenparameter
maschinengrenze_x = 370
maschinengrenze_y = 350

@dataclass
class Movement:
    x: Optional[float] = None
    y: Optional[float] = None
    is_relative: bool = False

@dataclass
class PlotterStatus:
    is_homed: bool = False
    current_x: float = 0.0
    current_y: float = 0.0
    max_x: float = maschinengrenze_x
    max_y: float = maschinengrenze_y
    is_relative_mode: bool = True  # True: RELATIVE, False: ABSOLUTE

class LaserPlotterTx:
    def __init__(self, port='/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.status = PlotterStatus()
        self.ser = None
        self.movement_history: List[Movement] = []

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Verbindung hergestellt: {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"Verbindungsfehler: {e}")
            return False

    def set_mode(self, is_relative: bool):
        """
        Setzt den Modus explizit auf RELATIV oder ABSOLUT.
        :param is_relative: True für RELATIV, False für ABSOLUT.
        """
        self.status.is_relative_mode = is_relative
        mode = "RELATIVE" if is_relative else "ABSOLUTE"
        print(f"Koordinatenmodus auf {mode} gesetzt.")

    def parse_movement(self, command: str, privileged: bool = False) -> Optional[Movement]:
        parts = command.upper().split()
        x_move = None
        y_move = None
        is_relative = self.status.is_relative_mode

        # Extrahiere Bewegungsbefehle
        for part in parts:
            if part.startswith('X'):
                try:
                    x_move = float(part[1:])
                except ValueError:
                    print(f"Warnung: Ungültiger X-Wert: {part}")
            elif part.startswith('Y'):
                try:
                    y_move = float(part[1:])
                except ValueError:
                    print(f"Warnung: Ungültiger Y-Wert: {part}")

        # Berechne neue Position
        new_x = self.status.current_x + (x_move or 0) if is_relative else (x_move or self.status.current_x)
        new_y = self.status.current_y + (y_move or 0) if is_relative else (y_move or self.status.current_y)

        # Grenzen prüfen (nur wenn nicht privilegiert)
        if not privileged and (new_x > self.status.max_x or new_y > self.status.max_y or new_x < 0 or new_y < 0):
            print(f"Bewegung würde außerhalb der Grenzen führen! Aktuell: X={self.status.current_x}, Y={self.status.current_y}, Ziel: X={new_x}, Y={new_y}")
            return None  # Keine gültige Bewegung

        return Movement(x_move, y_move, is_relative)

    def update_position(self, movement: Movement):
        """
        Aktualisiert die Position basierend auf der Bewegung und dem aktuellen Modus (relativ/absolut).
        """
        if movement.is_relative:  # Relative Bewegung
            if movement.x is not None:
                self.status.current_x += movement.x
            if movement.y is not None:
                self.status.current_y += movement.y
        else:  # Absolute Bewegung
            if movement.x is not None:
                self.status.current_x = movement.x
            if movement.y is not None:
                self.status.current_y = movement.y

        # Grenzen prüfen
        self.status.current_x = max(0, min(self.status.current_x, self.status.max_x))
        self.status.current_y = max(0, min(self.status.current_y, self.status.max_y))

        self.movement_history.append(movement)

    def send_gcode(self, command: str, privileged: bool = False) -> str:
        if not self.ser:
            raise Exception("Keine Verbindung zum Laserplotter")

        print(f"Prüfe: {command}")
        movement = self.parse_movement(command, privileged)

        if movement is None:
            return "Befehl abgelehnt: Bewegung außerhalb der Grenzen." if not privileged else "Fehler bei privilegiertem Befehl."

        # Sende den G-Code-Befehl direkt
        print(f"Sende: {command}")
        self.ser.write(f"{command}\n".encode())
        time.sleep(0.1)
        response = self.ser.readline().decode().strip()

        # Aktualisiere Position (wenn nicht privilegiert)
        if movement and not privileged:
            self.update_position(movement)
            print(f"Aktualisierte Position: X={self.status.current_x:.2f}, Y={self.status.current_y:.2f}")

        return response

    def home(self) -> bool:
        try:
            self.send_gcode("$X", privileged=True)
            self.movement_history = []
            self.set_mode(True)  # Setze RELATIVEN Modus für Homing

            # Home X-Achse
            self.send_gcode("G91 X10 F100", privileged=True)
            self.send_gcode("G91 X-370 F1000", privileged=True)
            self.send_gcode("G92 X0", privileged=True)
            time.sleep(5)

            # Home Y-Achse
            self.send_gcode("G91 Y10 F100", privileged=True)
            self.send_gcode("G91 Y-300 F1000", privileged=True)
            self.send_gcode("G92 Y0", privileged=True)

            self.set_mode(False)  # Optional: Zurück zu ABSOLUT, falls gewünscht

            self.status.current_x = 100  # Offset für die Düse
            self.status.current_y = 65
            self.status.is_homed = True
            return True
        except Exception as e:
            print(f"Homing-Fehler: {e}")
            return False

    def get_position_and_status(self) -> tuple:
        return (self.status.current_x, self.status.current_y, "RELATIVE" if self.status.is_relative_mode else "ABSOLUTE")

    def close_connection(self):
        if self.ser:
            self.ser.close()
            print("Serielle Verbindung geschlossen.")

def lasercommander():
    plotter = LaserPlotterTx()
    try:
        if not plotter.connect():
            return None

        if not plotter.status.is_homed:
            print("Führe Homing durch...")
            if not plotter.home():
                print("Homing fehlgeschlagen!")
                return None

        return plotter

    except Exception as e:
        print(f"Fehler: {e}")
        return None

if __name__ == "__main__":
    lasercommander()
