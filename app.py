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
# from License.padocr import PaddleOcr
class app:
    def __init__(self):
        pass

def draw(image, x1, y1, x2, y2, text="", color=(0, 255, 0), thickness=3, font_path="IBMPlexSansArabic-Regular.ttf", font_size=50):
    # Convert PIL image to NumPy array
    image = np.array(image)
    
    # Draw rectangle using OpenCV
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Convert NumPy array back to PIL image for text rendering
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    
    # Load the font
    font = ImageFont.truetype(font_path, font_size)
    
    # Draw text if provided
    if text != "":
        text_position = (x1, y1 - font_size)  # Adjust text position to be above the rectangle
        draw.text(text_position, text, font=font, fill=color)
        
    # Convert back to NumPy array
    image = np.array(image_pil)
    
    # Convert back to PIL image
    image = Image.fromarray(image)
    
    return image


    
def showimg(image):
    image = np.array(image)
    plt.imshow(image)
    plt.show()

# hello from m7dashraf
if __name__ == '__main__':
    frame_path = os.path.join(os.getcwd(), 'images', 'taxe4.jpg')
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
        # print("Plate Cord: ", platecord)

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
        # ocr = PaddleOcr()
        # license = ocr.recog(plate)
        print(license)
        for i in license:
            print(i)
        # print(f"License{0} ===== > ",license)
        # draw box arround Plate and write License
        frame = draw(frame, 
                cord[0] + platecord[0], # x1
                cord[1] + platecord[1], # y1
                cord[0] + platecord[2], # x2
                cord[1] + platecord[3], # y2
                text=''.join(license),
                color=(255, 0, 5)
            )

    showimg(frame)

