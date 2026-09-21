import os 
import time
import sys
import threading

import cv2 as cv
import open3d as o3d
import numpy as np 

from src.align_frames import point_to_point
from src.generate_mesh import generate_mesh

from pyorbbecsdk import *
from datetime import datetime, date
from src.pipeline_settings import apply_sdk_filters, TemporalFilter
# from rembg import new_session, remove
from pathlib import Path
from queue import Queue
from src import Turntable
# from testing.utilities.threshold_filter import threshold_filter
from utilities.utils import frame_to_bgr_image

np.set_printoptions(threshold=sys.maxsize)

PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 900  # 20mm
MAX_DEPTH = 1800  # 10000mm

#Remoção de clarões
#Source: https://stackoverflow.com/questions/72514384/how-can-i-remove-the-bright-glare-regions-in-image
def remove_bright_glare(img):
    hh, ww = img.shape[:2]

    lower = (220,220,220)
    upper = (300,300,300)
    thresh = cv.inRange(img, lower, upper)

    print("Applying threshold to remove bright glare")
    
    # apply morphology close and open to make mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel, iterations=1)

    print("Morphing threshold")
    
    # floodfill the outside with black
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = morph.copy()
    mask = cv.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]

    print("Floodfilling mask")
    
    return cv.inpaint(img, mask, 15, cv.INPAINT_TELEA), cv.inpaint(img, mask, 15, cv.INPAINT_NS)


def generate_frames(counter, timestamps, begin_t, spdir, path, frame_data, has_color_sensor, temporal_filter, save_queue):

    def process_depth_data(scale, depth_data):    

        mask = (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH)
        
        depth_data = np.where(mask, depth_data, 0)
        
        depth_m = depth_data.astype(np.float32) * scale
        
        # Apply temporal filtering
        depth_data = temporal_filter.process(depth_m)
        
        return depth_data
    

    counter[0] += 1
    counter_time = f"{int(counter[0]):02d}"

    elapsed = (time.time() - begin_t[0]) if begin_t[0] is not None else None
    if elapsed is not None:
        timestamps[counter_time] = elapsed
    else:
        print(f"[WARN] Frame {counter_time} capturado antes de begin_t ser setado -- sem timestamp valido")
        
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(path, f"{session_name}_frame_{counter_time}.npz")

    width = frame_data["depth_width"]
    height = frame_data["depth_height"]
    scale = frame_data["scale"]
    camera_param = frame_data["camera_param"]

    depth_original = frame_data["depth_data"].reshape((frame_data["depth_height"], frame_data["depth_width"])) 
    
    processed_depth = process_depth_data(scale, depth_original)
                
    if frame_data["color_data"].shape[:2] != processed_depth.shape:
        color_image = cv.resize(frame_data["color_data"], (width, height))
    
    save_queue.put((npz_path, {
        "color":     frame_data["color_data"],
        "depth_raw": depth_original,
        "scale":     scale,
        "fx": camera_param.rgb_intrinsic.fx, 
        "fy": camera_param.rgb_intrinsic.fy,
        "cx": camera_param.rgb_intrinsic.cx, 
        "cy": camera_param.rgb_intrinsic.cy
    }))

    
def mask_creation(depth_raw):
    depth_image = depth_raw.copy()
    
    mask_range = (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH)
    depth_data_masked = np.where(mask_range, depth_image, 0)
    depth_data_masked = depth_image.astype(np.uint16)
    
    depth_data_masked = np.where(mask_range, 255, 0).astype(np.uint8)
    
    mask_final = (depth_data_masked > 127).astype(np.uint8) * 255
    
    return mask_final
    
    
def only_person_mask(color_image, session):
    
    output = remove(color_image, session=session)

    if output.shape[2] == 4:
        print("if output.shape[2] == 4:")
        mask_rembg = np.array(output)[:, :, 3]
    else:
        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        _, mask_rembg = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    
    print("Thresholding body")
    
    mask_final = (mask_rembg > 127).astype(np.uint8) * 255
    
    return mask_final


def mask_refining(mask_raw):
    print(" Refining mask ")
    mask = mask_raw.copy()

    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
    print(" Closed holes ")

    mask = cv.GaussianBlur(mask, (5, 5), 0)
    print(" Smoothed edges ")

    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    print(" Binarized mask ")
    
    return mask


def applying_depth_mask(depth_raw, mask):
    print(" Applying mask in depth image ")
    depth_masked = depth_raw.copy()
    depth_masked[mask == 0] = 0

    original_valid = np.count_nonzero(depth_raw > 0)
    masked_valid   = np.count_nonzero(depth_masked > 0)
    removed        = original_valid - masked_valid

    print(f" Applied mask")
    print(f" Valid pixels before :  {original_valid:,}")
    print(f" Valid pixels after : {masked_valid:,}")
    print(f" Removed: {removed:,} ({100*removed/original_valid:.1f}%)")

    return depth_masked


def filtering_depth(depth_raw, color_image, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, aggressive_filtering=True):
    depth = depth_raw.astype(np.float32)

    mask_valid = (depth_raw > min_depth) & (depth_raw < max_depth) & (depth_raw > 0)
    depth[~mask_valid] = np.nan
    depth[depth == 0] = np.nan
    print(f" Pixels in valid range ({min_depth}-{max_depth}mm): {np.count_nonzero(mask_valid):,}")

    depth_for_filter = depth.copy()
    depth_for_filter[np.isnan(depth_for_filter)] = 0

    try:
        bilateral_filtered = cv.ximgproc.jointBilateralFilter(
            color_image.astype(np.float32),
            depth_for_filter.astype(np.float32),
            d=7, sigmaColor=50, sigmaSpace=50 
        )
        print("  Joint Bilateral filter applied ")
    except (cv.error, AttributeError):
        bilateral_filtered = cv.bilateralFilter(
            depth_for_filter.astype(np.float32),
            d=7, sigmaColor=50, sigmaSpace=50
        )
        print(" (fallback)")
    
    # Between joint bilateral and median — detect and zero all the flying pixels
    depth_uint16 = bilateral_filtered.astype(np.uint16)
    depth_median_local = cv.medianBlur(depth_uint16, 5).astype(np.float32)
    flying_pixels = (np.abs(bilateral_filtered - depth_median_local) > 50) & mask_valid
    bilateral_filtered[flying_pixels] = 0
    mask_valid[flying_pixels] = False
    print(f" Flying pixels removed: {flying_pixels.sum():,}")

    depth_filtered = bilateral_filtered.copy()

    depth_filtered = cv.bilateralFilter(
        depth_for_filter.astype(np.float32),
        d=7, sigmaColor=50, sigmaSpace=50
    )
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    depth_filtered = cv.erode(depth_filtered, kernel, iterations=3)

    return depth_filtered, bilateral_filtered, depth_for_filter, mask_valid
    

def process_npz(path):
    prev = time.time()
    print(f"tempo inicial de processamento:{prev} segundos")

    def generating_point_cloud(color_image, depth_filtered, scale, fx, fy, cx, cy):
        print(" Generating the point cloud... ")
        height, width = depth_filtered.shape

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z    = depth_filtered * scale

        valid_mask  = z > 0
        valid_count = np.count_nonzero(valid_mask)
        print(f"   Valid pixels for point cloud: {valid_count:,}")

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)[valid_mask]
        colors = color_image[valid_mask][:, [2, 1, 0]]  # BGR → RGB
        
        # ← aqui
        brilho = colors[:, 0].astype(float) + colors[:, 1].astype(float) + colors[:, 2].astype(float)
        mask_preto = brilho > 60 
        
        points = points[mask_preto]
        colors = colors[mask_preto]

        return points, colors


    def save_results(base_name, mask, points, colors, depth_raw, depth_masked, mask_raw, depth_filtered, bilateral_filtered, depth_for_filter, color_image, output_dir="./point_clouds/{}"):
        def save_custom_ply(filename):
            with open(filename, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                # Dados de pontos e cores
                for point, color in zip(points, colors):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
                    f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

        os.makedirs(output_dir, exist_ok=True)

        ply_path = os.path.join(output_dir, f"{base_name}_clean.ply")
        save_custom_ply(ply_path)
        print(f"✅ Point cloud saved: {ply_path}")
        
        # color_image_shadow_removed_path = os.path.join(output_dir, f"{base_name}_color_shadow_removed.png")
        # cv.imwrite(color_image_shadow_removed_path, color_image_shadow_removed)
        # print(f" Mask saved: {color_image_shadow_removed_path}")
        
        color_image_path = os.path.join(output_dir, f"{base_name}_color_image.png")
        cv.imwrite(color_image_path, color_image)
        print(f"Saved mask: {color_image_path}")
        
        mask_raw_path = os.path.join(output_dir, f"{base_name}_mask_raw.png")
        cv.imwrite(mask_raw_path, mask_raw)
        print(f"Saved mask: {mask_raw_path}")

        depth_vis = cv.normalize(depth_filtered, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        depth_path = os.path.join(output_dir, f"{base_name}_depth_filtered.png")
        cv.imwrite(depth_path, depth_vis)
        print(f"Saved filtered depth: {depth_path}")
        
        # bilateral_path = os.path.join(output_dir, f"{base_name}_bilateral_filtered.png")
        # cv.imwrite(bilateral_path, bilateral_filtered)
        # print(f"Saved filtered depth: {bilateral_path}")
        
        # depth_for_filter_path = os.path.join(output_dir, f"{base_name}_depth_for_filter.png")
        # cv.imwrite(depth_for_filter_path, depth_for_filter)
        # print(f"Saved filtered depth: {depth_for_filter_path}")
        
        depth_raw_path = os.path.join(output_dir, f"{base_name}_depth_raw.png")
        cv.imwrite(depth_raw_path, depth_raw)
        print(f"Saved filtered depth: {depth_raw_path}")
        
        depth_masked_path = os.path.join(output_dir, f"{base_name}_depth_masked.png")
        cv.imwrite(depth_masked_path, depth_masked)
        print(f"Saved depth masked: {depth_masked_path}")

        preview = color_image.copy()
        preview[mask == 0] = [0, 0, 0]
        preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
        cv.imwrite(preview_path, preview)
        print(f"Saved preview: {preview_path}")


    #Para usar o REMBG com o modelo BRIA descomentar abaixo
    # begin_session = time.time()
    #session = new_session("bria-rmbg", providers=["CPUExecutionProvider"])
    # end_session = time.time() - begin_session
    # print(f"\nend session time:{end_session}")
    # print(session.inner_session.get_providers())
    
    npz_files = sorted(Path(path).glob("*.npz"))
    print(f"{len(npz_files)} files found\n")
    
    for i, filepath in enumerate(npz_files):
        if "angles" not in filepath.name:
            print(f"\n{'='*60}")
            print(f"📂 {filepath.name}")
            print(f"{'='*60}")

            data        = np.load(filepath)
            depth_raw   = data["depth_raw"]
            color_image = data["color"]
                            
            scale = float(data.get("scale", 1.0))
            fx, fy = float(data["fx"]), float(data["fy"])
            cx, cy = float(data["cx"]), float(data["cy"])
            
            #inpaint_telea ,_                                       = remove_bright_glare(color_image) 
            mask_raw                                                = mask_creation(depth_raw)
            #mask_raw                                                = only_person_mask(color_image, session)   
            mask                                                    = mask_refining(mask_raw)
            depth_masked                                            = applying_depth_mask(depth_raw, mask)
            depth_filtered, bilateral_filtered, depth_for_filter, _ = filtering_depth(depth_masked, color_image)
            points, colors                                          = generating_point_cloud(color_image, depth_filtered, scale, fx, fy, cx, cy)

            base_name = filepath.stem
            save_results(base_name, mask, points, colors, depth_raw, depth_masked, mask_raw, depth_filtered, bilateral_filtered, depth_for_filter, color_image, path)

            actual1 = time.time() - prev
            print(f"geração de imagens e nuvem de pontos em:{actual1} segundos")

                
#Para geração de point clouds diretamente
def generate_point_cloud_directly(spdir, current_date, frames, color_frame, align_filter, has_color_sensor, point_cloud_filter):
    while True:
        # #Below there is the Orbbec's implementation
        frame = align_filter.process(frames)
        #scale = depth_frame.get_depth_scale()
        #point_cloud_filter.set_position_data_scaled(scale)

        point_format = OBFormat.RGB_POINT if has_color_sensor and color_frame is not None else OBFormat.POINT
        point_cloud_filter.set_create_point_format(point_format) 
                    
        pc_frame = point_cloud_filter.process(frame)
        if pc_frame is None: 
            continue
        
        save_point_cloud_to_ply(os.path.join(save_points_dir, "point_cloud.ply"), point_cloud_frame)
        

def frames_pipeline(depth_frame, timestamps, begin_t, frame_queue, save_queue, spdir, path, pipeline, has_color_sensor, origin_reached, align_filter, sdk_filters):
    counter         = [0]
    temporal_filter = TemporalFilter(alpha=0.5)
    
    def worker_save():
        while True:
            item = save_queue.get()
            if item is None:
                break
            npz_path, data = item
            np.savez(npz_path, **data)
            save_queue.task_done()

    t_save = threading.Thread(target=worker_save, daemon=True)
    t_save.start()
    
    while not origin_reached.is_set():
        try:
            frames_raw = frame_queue.get(timeout=1.0)
        except:
            continue
        
        depth_frame   = frames_raw.get_depth_frame()
        depth_frame   = apply_sdk_filters(depth_frame, sdk_filters)
        aligned_frame = align_filter.process(frames_raw)
        if aligned_frame is None:
            continue
        
        aligned_frameset = aligned_frame.as_frame_set()
        aligned_depth_frame = aligned_frameset.get_depth_frame()
        aligned_color_frame = aligned_frameset.get_color_frame()
        
        if aligned_depth_frame is None or aligned_color_frame is None:
            continue
        
        depth_format = aligned_depth_frame.get_format()
        if depth_format != OBFormat.Y16:
            print("depth format is not Y16")
            continue
        
        color_image = frame_to_bgr_image(aligned_color_frame).copy()
        depth_data  = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16).copy()
        
        frame_data = {
            "color_data":   color_image,
            "depth_data":   depth_data,
            "depth_width":  aligned_depth_frame.get_width(),
            "depth_height": aligned_depth_frame.get_height(),
            "scale":        aligned_depth_frame.get_depth_scale(),
            "camera_param": pipeline.get_camera_param()
        }
                
        #Pre-processed version from myself
        generate_frames(counter, timestamps, begin_t, spdir, path, frame_data, has_color_sensor, temporal_filter, save_queue)
    
    save_queue.put(None)
    t_save.join()
    print(" [SUCCESS] All frames saved successfully")