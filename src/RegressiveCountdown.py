import cv2 as cv
from datetime import datetime #import time


class RegressiveCountdown:
    def __init__(self, duration = 5):
        self.duration = duration
        self.start_time = None
        self.active = False
    
    def start(self):
        self.start_time = datetime.now()
        self.active = True
        
    def reset(self):
        self.start_time = datetime.now()
        self.active = True
        
    def stop(self):
        self.active = False
        self.start_time = None
       
    def overlay(self, frame, win_name, color=(255,0,0)):
        if not self.active or self.start_time is None:
            return frame, False

        diff = (datetime.now() - self.start_time).total_seconds()
        remaining = int(self.duration - diff)

        if remaining <= 0:
            remaining = 1
            finished = True
        else:
            finished = False

        h, w = frame.shape[:2]
        org = (w // 2 - 20, h // 2)

        cv.putText(frame,str(remaining),org,cv.FONT_HERSHEY_SIMPLEX,2,color,4,cv.LINE_AA)

        if(finished):
            self.stop()

        return frame, finished