import cv2
import numpy as np
from typing import Tuple, Optional

class SyntheticRulerGenerator:
    """
    A class to generate synthetic ruler images for training and validation.
    """

    def __init__(self, width: int = 512, height: int = 512):
        """
        Initializes the generator with default image dimensions.
        """
        self.width = width
        self.height = height

    def generate_ruler(
        self,
        ruler_thickness: int = 10,
        tick_width: int = 1,
        tick_length: int = 20,
        tick_spacing: int = 50,
        rotation_angle: float = 0.0,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        ruler_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Generates a synthetic ruler image with the specified properties.
        """
        img = np.full((self.height, self.width, 3), background_color, dtype=np.uint8)

        # Create a blank canvas to draw the ruler on
        ruler_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the ruler body
        ruler_y = self.height // 2
        cv2.line(ruler_canvas, (0, ruler_y - ruler_thickness // 2), (self.width, ruler_y - ruler_thickness // 2), ruler_color, tick_width)
        cv2.line(ruler_canvas, (0, ruler_y + ruler_thickness // 2), (self.width, ruler_y + ruler_thickness // 2), ruler_color, tick_width)

        # Draw the ticks
        for x in range(0, self.width, tick_spacing):
            cv2.line(ruler_canvas, (x, ruler_y - tick_length // 2), (x, ruler_y + tick_length // 2), ruler_color, tick_width)

        # Rotate the ruler
        if rotation_angle != 0.0:
            center = (self.width // 2, self.height // 2)
            rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            ruler_canvas = cv2.warpAffine(ruler_canvas, rot_mat, (self.width, self.height))

        # Add the ruler to the background
        img[ruler_canvas > 0] = ruler_canvas[ruler_canvas > 0]

        return img

if __name__ == '__main__':
    generator = SyntheticRulerGenerator()
    ruler_image = generator.generate_ruler(rotation_angle=15)
    cv2.imwrite("synthetic_ruler.png", ruler_image)
    print("Synthetic ruler image saved to synthetic_ruler.png")
