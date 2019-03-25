import cv2
import numpy as np

class Generator():

    def __init__(self):
        self.base_dir=''
        self.


    def get_all_img(self):
        return self._image_ids

    def read_img(self,img_name):
        img_path =self._image_file.format(bytes.decode(img_name))
        img=cv2.imread(img_path)
        return img.astype(np.float32)

    def detections(self, img_name):
        detections = self._detections[bytes.decode(img_name)]
        return detections
