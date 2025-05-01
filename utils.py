import numpy as np

if not hasattr(np, 'infty'):
    setattr(np, 'infty', np.inf)

import trimesh
import pyrender


def visualize_model(path: str) -> None:
 
    mesh = trimesh.load(path)

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))


    pyrender.Viewer(scene, use_raymond_lighting=True)
