import numpy as np
import os 
import re
from PIL import Image
import genbrain_model_3dot as model

def collect_frames(directory="../data/", file_pattern="frame-*.png"):
    frame_regex = re.compile(r"frame-(\d+)\.png")
    frames = []
    for filename in os.listdir(directory):
        match = frame_regex.match(filename)
        if match:
            frame_number = int(match.group(1))
            frames.append((frame_number, os.path.join(directory, filename)))
    frames.sort(key=lambda x: x[0])
    frame_paths = [path for _, path in frames]
    numpy_frames = []
    for path in frame_paths:
        with Image.open(path) as img:
            # this must be going downwards?
            im = img.convert("L").point(lambda p: 1 if p > 0 else 0)
            numpy_frames.append((np.transpose(np.flipud(np.array(im))) > 0).astype(int))
    return numpy_frames

def get_xyz(results):
    particles_per_step = results[1]
    xyz_vals = []
    for particles in particles_per_step:
        xyz_vals.append([model.xyz_point_cloud[p.choicemap["xyz"]] for p in particles])
    return xyz_vals

