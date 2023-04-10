import glob
import os
import cv2

class DataLoader():
    def __init__(self):
        self.images = []
        self.loaded = False


    def load_images(self, numDays, numCams):
        """
        load_images loads all files into a python list self.images

        Args:
            numDays (int): number of days in the MVOR dataset
            numCams (int): num of cameras in the MVOR dataset
        """

        MVOR_DIR = os.path.join(os.getcwd(),'mvor')
        for day_num in range(1, numDays + 1):
            for cam_num in range(1, numCams + 1):
                dir_path = os.path.join(MVOR_DIR, f'day{day_num}', f'cam{cam_num}', '*png')
                frames = glob.glob(dir_path)
                for frame in frames:
                    img_name = frame.split('/')[-1]
                    img_num, ext = img_name.split('.')
                    img_id = f'{day_num}00{cam_num}0{img_num}'
                    
                    image = cv2.imread(frame)
                    self.images.append(image)
        self.loaded = True

    
    def get_item(self, idx):
        """
        get_item indexes the list to put a image from the list, only if the images have
        already been loaded using load_images

        Args:
            index of the image we want to load
        """
        if self.loaded:
            return self.images[idx]
        raise Exception("DataLoader has not loaded imags") 