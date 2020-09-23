from mtcnn import MTCNN
import cv2
import os
from PIL import ImageDraw,Image
import matplotlib.pyplot as plt

path = r"D:\BaiduNetdiskDownload\CelebA\Img\img_celeba.7z\img_celeba"
file = os.listdir(path)
img = cv2.cvtColor(cv2.imread("1.png"), cv2.COLOR_BGR2RGB)

detector = MTCNN()
data = detector.detect_faces(img)
print(data)
img = Image.fromarray(img,"RGB")
draw = ImageDraw.Draw(img)
#draw.rectangle([138, 70, 206, 260],outline="red",width=2)

draw.ellipse((176,208,186,218),fill = (255, 0, 0),width=2)
draw.ellipse((295,212,305,222),fill = (255, 0, 0),width=2)
plt.imshow(img)
plt.show()