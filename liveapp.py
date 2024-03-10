# -*- coding: utf-8 -*-
import sys
import torch
import cv2
import cvzone
import numpy as np
import os
from PIL import Image,ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt

from Vehicle.Detection import detection
from Plate.Detection.detection import PlateDatection
from Vehicle.Detection import CutIm
from License.Recognition.recognition import Recognition

def draw(image, x1, y1, x2, y2, text="", color=(0, 255, 0), thickness=3):
    image = np.array(image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if text != "":
        text_position = (x1, y1 - 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 
    image = Image.fromarray(image)
    return image


def go2live():
    path = os.path.join(os.getcwd(), 'videos', 'Traffic in Cairo Egypt.mp4')
    cap = cv2.VideoCapture(path)

    cn = 0
    while True:
        cn += 1
        success, frame = cap.read()
        frame = Image.fromarray(frame)
        VehicleDetector = detection.VehicleDetection()

        cords_of_vehicles = VehicleDetector.detect(frame)

        for cord in cords_of_vehicles:
            # first step detect Vehicel
            vehicle = CutIm.CropPlate(frame, cord).crop()

            # second step detect Plate
            plateDetector = PlateDatection()
            platecord = plateDetector.detect(vehicle)
            print("Plate Cord: ", platecord)

            if platecord == ():
                continue

            #crop the part that cotain Plate
            plate = CutIm.CropPlate(vehicle, platecord).crop()

            # third step Recoginize the License
            recognizer = Recognition()
            license = recognizer.recog(plate) 
            
            # draw box arround Vehicel
            frame = draw(frame, cord[0], cord[1], cord[2], cord[3])

            # draw box arround Plate and write License
            frame = draw(frame, 
                    cord[0] + platecord[0], # x1
                    cord[1] + platecord[1], # y1
                    cord[0] + platecord[2], # x2
                    cord[1] + platecord[3], # y2
                    text=license,
                    color=(255, 0, 5)
                )
        frame = np.array(frame)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        
        if cn == 10:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    go2live()