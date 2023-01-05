import cv2
import glob
from tqdm import tqdm

DATA_PATH = "data/rgb_frames"

for folder in tqdm(glob.glob(DATA_PATH + "/*")):
    images = glob.glob(folder + "/*.jpg")
    images = sorted(images, key=lambda x: int(x.split("\\")[-1].split("_")[1].split(".")[0]), reverse=False)
    img_array = []
    for image in images:
        img = cv2.imread(image)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    video_name = folder.split("\\")[-1]
    out = cv2.VideoWriter("videos/" + video_name + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()