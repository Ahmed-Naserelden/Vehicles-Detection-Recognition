# -*- coding: utf-8 -*-
import sys
import torch
import cv2
import cvzone
import numpy as np
from PIL import Image,ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from Vehicle.Detection import detection
from Plate.Detection.detection import PlateDatection
from Vehicle.Detection import CutIm
from License.Recognition.recognition import Recognition

class app:
    def __init__(self):
        pass

def draw(image, x1, y1, x2, y2, text="", color=(0, 255, 0), thickness=3):
    image = np.array(image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if text != "":
        text_position = (x1, y1 - 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 
    image = Image.fromarray(image)
    return image


    
def showimg(image):
    image = np.array(image)
    plt.imshow(image)
    plt.show()

# hello from m7dashraf
if __name__ == '__main__':
    frame_path = os.path.join(os.getcwd(), 'images', 'mohamed1.jpeg')
    frame = Image.open(frame_path)
    VehicleDetector = detection.VehicleDetection()

    cords_of_vehicles = VehicleDetector.detect(frame)

    for cord , cls_ in cords_of_vehicles:
        # first step detect Vehicel
        vehicle = CutIm.CropPlate(frame, cord).crop()
        # showimg(vehicle)

        # second step detect Plate
        plateDetector = PlateDatection()
        platecord = plateDetector.detect(vehicle)
        print("Plate Cord: ", platecord)

        # draw box arround Vehicel
        frame = draw(frame, cord[0], cord[1], cord[2], cord[3])

        if platecord == ():
            continue

        #crop the part that cotain Plate
        plate = CutIm.CropPlate(vehicle, platecord).crop()
        # showimg(plate)

        # third step Recoginize the License
        recognizer = Recognition()
        license = recognizer.recog(plate) 
        
        # draw box arround Vehicel
        # frame = draw(frame, cord[0], cord[1], cord[2], cord[3])
        
        # draw box arround Plate and write License
        frame = draw(frame, 
                cord[0] + platecord[0], # x1
                cord[1] + platecord[1], # y1
                cord[0] + platecord[2], # x2
                cord[1] + platecord[3], # y2
                text=license,
                color=(255, 0, 5)
            )

    showimg(frame)

