import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

CLASS_NAMES = {0: 'alif', 1: 'baa', 2: 'daal', 3: 'eight', 4: 'ein', 5: 'faa', 6: 'five', 7: 'four', 8: 'geem', 9: 'haa', 10: 'kaaf', 11: 'laam', 12: 'meem', 13: 'nine', 14: 'noon', 15: 'one', 16: 'qaaf', 17: 'raa', 18: 'saad', 19: 'seen', 20: 'seven', 21: 'six', 22: 'taa', 23: 'three', 24: 'two', 25: 'waaw', 26: 'yaa'}

# Define a custom key function for sorting
def custom_key(coord):
    # Sort by x1 first, then y1, x2, and y2
    return (coord[0], coord[1], coord[2], coord[3])

class Recognition:

    def __init__(self):
        self.model = YOLO('../models/best.pt')
        self.result = []
        # print(self.model.names)

    def recog(self, img):

        result = self.model('../../images/afg.jpg')
        
        for r in result:
            boxes = r.boxes

            for box in boxes:
            
                # object cordinate 
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # to get label of object
                cls_ = CLASS_NAMES[int(box.cls[0])]


                # save object cordinate and its label into result
                self.result.append((x1, y1, x2, y2, cls_))
                
                # print(x1, y1, x2, y2, " -> ", cls_)
        

        # return license of Vehicel in order from left to wirte
        srt_li =  sorted(self.result, key=custom_key)

        license = [ele[-1] for ele in srt_li]
        return license
        

if __name__ == '__main__':
    rec = Recognition()
    result = rec.recog('../../images/afg.jpg')

    # print each char in license
    for char in result:
        print(char, end=" ")
    