from rembg import remove
from PIL import Image
import trimesh
import numpy as np
import cv2
import os
from utils import visualize_model

def process_image_to_3d(image_path):
    img = Image.open(image_path)
    fg = remove(img)  # Background removal
    fg.save("output/clean.png")

    img_cv = cv2.imread("output/clean.png", cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_cv, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_map = np.zeros_like(img_cv, dtype=np.float32)

    cv2.drawContours(height_map, contours, -1, 255, thickness=cv2.FILLED)

    # Simulate extrusion
    mesh = trimesh.creation.extrude_polygon(trimesh.path.polygons.projected(height_map), height=5.0)

    mesh.export("output/model_from_image.stl")
    mesh.show()

    print("3D model saved as output/model_from_image.stl")
    visualize_model("output/model_from_image.stl")