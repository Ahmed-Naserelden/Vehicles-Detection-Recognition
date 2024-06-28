import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from PIL import Image
# Initialize the OCR model with Arabic language support
ocr = PaddleOCR(use_angle_cls=True, lang="ar")

# Specify the path to the image

class PaddleOcr():

    def __init__(self):
        # Initialize the PaddleOCR with Arabic language and angle classification
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ar")
        
    def recog(self, img):


        # img_pil = Image.open(img).convert("RGB")
        # Convert PIL image to OpenCV image
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Perform OCR on the image
        result = self.ocr.ocr(img, cls=True)
        
        # Print the entire OCR result
        print("OCR Results:", result)

        # Extract the text, boxes, and scores from the result and print them
        part1, part2 = "", ""
        for idx, line in enumerate(result):
            for box in line:
                text = box[1][0]
                confidence = box[1][1]
                # box_coords = box[0]
                part1 = part2
                part2 = text
                # print(f"Line {idx + 1}:")
                # print(f"  Text: {text}")
                # print(f"  Confidence: {confidence}")
                # print(f"  Box Coordinates: {box_coords}")
                
            """
                # Draw the bounding box on the image
                cv2.polylines(img, [np.array(box_coords).astype(np.int32)], True, (0, 255, 0), 2)
                # Put the text above the bounding box
                # Calculate text size to determine the position for the text
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int(box_coords[0][0])
                text_y = int(box_coords[0][1]) - 10  # Adjust this to place the text above the box
                # Ensure the text is not placed out of the image boundaries
                text_y = max(text_height, text_y)
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                """
        """
        # Convert the image color from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display the image using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        """
        return part1 + " " + part2

if __name__ == '__main__':
    image_path = '../images/ars.jpeg'
    img = cv2.imread(image_path)


    ocr = PaddleOcr()
    license = ocr.recog(img)
    print(license)