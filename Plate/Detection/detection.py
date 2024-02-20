import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

import sys
# sys.path.append('')
from CutIm import CropPlate

class PlateDatection:
    def __init__(self):
        self.model = YOLO('../models/best.pt')
        self.cord = []

    def detect(self, img):
        result = self.model(img)


        print("Results: ")
        # print(result[0])
        for r in result:
            boxes = r.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                self.cord.append((x1, y1, x2, y2))

        return self.cord[0]

if __name__ == '__main__':

    image_path = '../../images/taxe.jpg'
    output_image_path = '../../images/Plates/plate1.jpg'
    img = Image.open(image_path)

    model = PlateDatection()
    cord = model.detect(image_path)
    print(cord)
    
    cropped_image = CropPlate(img, cord).crop()
    
    # cropped_image = img.crop(cord)
    # cropped_image.save(output_image_path)
    # cropped_image = Image.open(output_image_path)
    cropped_image.show()

    # plt.imshow(cropped_image)
    # plt.show()

