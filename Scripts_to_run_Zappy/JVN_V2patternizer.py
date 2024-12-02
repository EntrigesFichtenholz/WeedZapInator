import math

class CirclePatternizer:
    """
    Generates concentric circles from the outermost to the innermost.

    :param radius: Initial radius of the outermost circle.
    :param radius_step: Step size to reduce the radius for each inner circle.
    :param steps: Number of segments to divide each circle into.
    """
    def __init__(self, radius=50, radius_step=5, steps=36):
        self.max_radius = radius  # Compatible with the main function's parameter naming
        self.radius_step = radius_step
        self.steps = steps

    def generate_gcode(self):
        """
        Generate G-code commands for concentric circles.
        """
        gcode_commands = []
        current_radius = self.max_radius

        # Generate circles from outermost to innermost
        while current_radius > 0:
            # Generate one full circle
            for i in range(self.steps + 1):  # +1 to close the circle
                angle = 2 * math.pi * i / self.steps
                x = current_radius * math.cos(angle)
                y = current_radius * math.sin(angle)
                gcode_commands.append(f"G1 X{round(x, 2)} Y{round(y, 2)} F1000")

            # Reduce radius for the next circle
            current_radius -= self.radius_step


        return gcode_commands

class SpiralPatternizer:
    def __init__(self, max_radius=50, radius_step=2, steps=36):
        """
        Generiert eine einfache Spirale mit linearer Zunahme
        
        :param max_radius: Maximaler Radius
        :param radius_step: Schrittweise Vergrößerung des Radius
        :param steps: Anzahl der Schritte pro Kreisumrundung
        """
        self.max_radius = max_radius
        self.radius_step = radius_step
        self.steps = steps

    def generate_gcode(self):
        gcode_commands = []
        current_radius = 0
        
        while current_radius < self.max_radius:
            for i in range(self.steps):
                angle = 2 * math.pi * i / self.steps
                x = current_radius * math.cos(angle)
                y = current_radius * math.sin(angle)
                gcode_commands.append(f"G1 X{round(x, 2)} Y{round(y, 2)} F1000")
            
            current_radius += self.radius_step
        
        return gcode_commands

class GridPatternizer:
    def __init__(self, width=100, height=100, step_size=10):
        """
        Generiert ein Rastermuster
        
        :param width: Breite des Rasters
        :param height: Höhe des Rasters
        :param step_size: Schrittgröße zwischen Punkten
        """
        self.width = width
        self.height = height
        self.step_size = step_size

    def generate_gcode(self):
        gcode_commands = []
        
        # Horizontale Linien
        for y in range(0, self.height + 1, self.step_size):
            gcode_commands.append(f"G1 X0 Y{y} F1000")
            gcode_commands.append(f"G1 X{self.width} Y{y} F1000")
        
        # Vertikale Linien
        for x in range(0, self.width + 1, self.step_size):
            gcode_commands.append(f"G1 X{x} Y0 F1000")
            gcode_commands.append(f"G1 X{x} Y{self.height} F1000")
        
        return gcode_commands

class RandomPatternizer:
    def __init__(self, width=100, height=100, points=50):
        """
        Generiert zufällige Punkte innerhalb eines Bereichs
        
        :param width: Breite des Bereichs
        :param height: Höhe des Bereichs
        :param points: Anzahl der zufälligen Punkte
        """
        import random
        self.width = width
        self.height = height
        self.points = points
        self.random = random

    def generate_gcode(self):
        gcode_commands = []
        
        for _ in range(self.points):
            x = self.random.uniform(0, self.width)
            y = self.random.uniform(0, self.height)
            gcode_commands.append(f"G1 X{round(x, 2)} Y{round(y, 2)} F1000")
        
        return gcode_commands
