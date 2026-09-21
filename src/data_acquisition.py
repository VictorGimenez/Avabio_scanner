import time
import os
import glob
import threading
import traceback
import json

import numpy as np

from src.mesh_manipulation import cropping
from src.generate_mesh import generate_mesh
from src.file_handling import check_and_mkdir


def rotate_complete(turntable_event: threading.Event, turntable, timestamps, begin_t, origin_reached, path):
    def track():
        turntable.waiting_origin()
        origin_reached.set()
    
    turntable.rotate()
    # begin_t[0] = time.time()
    
    t = threading.Thread(target=track, daemon=True)
    t.start()
    
    turntable_event.set()
    
    while not origin_reached.is_set():
        pass

    t_total = time.time() - begin_t[0]
    # angles = [(t / t_total) * 360 for t in timestamps]
    
    # angles = [a for a in angles if 0 <= a <= 360]
    
    # for i in range(1, len(angles)):
    #     diff = angles[i] - angles[i-1]
    #     print(f"Frame {i} → {i+1}: {diff:.1f}°")
        
    # npz_path = os.path.join(path, "angles.npy")
    # np.save(npz_path, np.array(angles))
    
    # for i, angle in enumerate(angles):
    #     print(f"  Frame {i+1}: {angle:.1f}°")
    
    # print(angles)
    
    angles_by_frame = {}
    print(f"timestamps:{timestamps}")
    for frame_key, t_elapsed in timestamps.items():
        angle = (t_elapsed / t_total) * 360
        if 0 <= angle <= 360:
            angles_by_frame[frame_key] = angle
        else:
            print(f"[WARN] Frame {frame_key} descartado (angulo fora de faixa: {angle:.1f})")

    json_path = os.path.join(path, "angles.json")
    with open(json_path, "w") as f:
        json.dump(angles_by_frame, f, indent=2)

    for k in sorted(angles_by_frame, key=int):
        print(f"  Frame {k}: {angles_by_frame[k]:.1f}°")