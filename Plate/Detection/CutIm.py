from PIL import Image

class CropPlate:
    def __init__(self, input_image, crop_box):
        self.img = input_image
        self.crop_box = crop_box

    def crop(self):
        # Crop the image from (x1, y1) to (x2, y2)
        cropped_image = self.img.crop(self.crop_box)
        return cropped_image

if __name__ == '__main__':
    pass