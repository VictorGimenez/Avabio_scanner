from datetime import datetime #import time
import cv2 as cv
import numpy as np
import time
import queue
import threading

from pyorbbecsdk import *
from utilities.utils import frame_to_bgr_image
from src.pipeline_settings import apply_sdk_filters
from rembg import remove

# from countdown import regressive_countdown


def make_transparent_overlay(COLOR_TO_REMOVE, depth_bgr):
    # Guarantee 4 channels
    depth_bgra = cv.cvtColor(depth_bgr, cv.COLOR_BGR2BGRA)

    # BGR masking
    mask = np.all(depth_bgra[:, :, :3] == COLOR_TO_REMOVE, axis=2)

    # Is there are forbidden colors alpha is zeroed
    depth_bgra[mask, 3] = 0

    return depth_bgra


def alpha_blend(fg, bg):
    if bg.shape[2] == 3:
        bg = cv.cvtColor(bg, cv.COLOR_BGR2BGRA)

    # Normalizing alpha foreground
    alpha = fg[:, :, 3:4].astype(float) / 255.0
    inv_alpha = 1.0 - alpha

    out = (alpha * fg[:, :, :3] + inv_alpha * bg[:, :, :3]).astype(np.uint8)
    out_alpha = (alpha * 255 + inv_alpha * bg[:, :, 3:4]).astype(np.uint8)

    return np.dstack([out, out_alpha])


def apply_thresh_filter(color_image, frames, align_filter, COLOR_TO_REMOVE, MIN_DEPTH=0, MAX_DEPTH=50000, ALERT_COLOR=(125, 0, 0), ALERT_TOLERANCE=30, ALERT_MIN_PIXELS=500):
    
    aligned_frame = align_filter.process(frames)
    if aligned_frame is None:
        return color_image, False
    
    aligned_frameset = aligned_frame.as_frame_set()
    aligned_depth_frame = aligned_frameset.get_depth_frame()
    if aligned_depth_frame is None:
        return color_image, False
    
    width = aligned_depth_frame.get_width()
    height = aligned_depth_frame.get_height()
    scale = aligned_depth_frame.get_depth_scale()
    
    depth_raw = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
    depth_mm = (depth_raw.astype(np.float32) * scale).astype(np.uint16)
    
    # Threshold
    mask = (depth_mm >= MIN_DEPTH) & (depth_mm <= MAX_DEPTH)
    depth_filtered = np.where(mask, depth_mm, 0).astype(np.uint16)

    # Normalização com escala FIXA (não mais NORM_MINMAX por frame)
    span = max(MAX_DEPTH - MIN_DEPTH, 1)
    depth_norm = np.clip(
        (depth_filtered.astype(np.float32) - MIN_DEPTH) / span * 255, 0, 255
    ).astype(np.uint8)
    depth_colored = cv.applyColorMap(depth_norm, cv.COLORMAP_JET)

    # --- Procura a cor alvo só dentro da região válida (mask) ---
    alvo  = np.array(ALERT_COLOR, dtype=np.int16)   # BGR!
    lower = np.clip(alvo - ALERT_TOLERANCE, 0, 255).astype(np.uint8)
    upper = np.clip(alvo + ALERT_TOLERANCE, 0, 255).astype(np.uint8)
    color_mask = cv.inRange(depth_colored, lower, upper)
    color_mask[~mask] = 0   # ignora qualquer coisa fora do threshold
    color_detected = bool(cv.countNonZero(color_mask) >= ALERT_MIN_PIXELS)

    # Removing yellow pixels
    depth_overlay = make_transparent_overlay(COLOR_TO_REMOVE, depth_colored)

    # Ensure same size
    if depth_overlay.shape[:2] != color_image.shape[:2]:
        depth_overlay = cv.resize(depth_overlay, (color_image.shape[1], color_image.shape[0]),
                                    interpolation=cv.INTER_NEAREST)
                    
    # Composing depth over color
    result = alpha_blend(depth_overlay, color_image)
    
    return result, color_detected
        

def display_camera(pipeline, has_color_sensor, sdk_filters, align_filter): 
    while True:
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            yield None, None, None  # ← yield None em vez de continue, para não travar o for
            continue
        
        color_frame = frames.get_color_frame()
        if has_color_sensor and color_frame is None:
            yield None, None, None
            continue
        
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            yield None, None, None
            continue
        
        # converting to RGB format
        if has_color_sensor and color_frame is not None:
            color_image = frame_to_bgr_image(color_frame)
        else:
            color_image = np.zeros((color_frame.get_height(), color_frame.get_width(), 3), dtype=np.uint8)
        
        if color_image is None:
            print("failed to convert frame to image")
            continue
        
        # if processed_color.shape[:2] != processed_depth.shape:
        #     processed_color = cv.resize(processed_color, (width, height))
        
        h, w = color_image.shape[:2]
        
        # # threshold
        # lower = (220,220,220)
        # upper = (300,300,300)
        # thresh = cv.inRange(color_image, lower, upper)
        
        # # apply morphology close and open for masking
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        # morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        # morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel, iterations=1)
        
        # # floodfill the outside with black
        # black = np.zeros([h + 2, w + 2], np.uint8)
        # mask = morph.copy()
        # mask = cv.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]
        
        # color_image = cv.inpaint(color_image, mask, 15, cv.INPAINT_TELEA)
        
        cy   = h // 2

        y_start = cy - 220
        y_end   = cy + 220
        
        result = color_image.copy()
        cv.line(result, (0, cy),      (w, cy),      (0, 255, 0), 2)  # centro — verde
        cv.line(result, (0, y_start), (w, y_start), (0, 0, 255), 5)  # limite superior — vermelho
        cv.line(result, (0, y_end),   (w, y_end),   (0, 0, 255), 5)  # limite inferior — vermelho

        # Draw a diagonal blue line with thickness of 5 px
        # cv.rectangle(result,(h+400,100),(h+400,200),(0,255,0),3)
        
        # color_image = np.zeros_like(color_image)  # black image
        #color_image[y_start:y_end, :] = color_image[y_start:y_end, :]  # Only valid band
        #color_image = remove(color_image, session=session)
        
        yield result, depth_frame, frames