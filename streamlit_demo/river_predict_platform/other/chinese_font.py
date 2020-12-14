import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import time


## Use simsum.ttc to write Chinese.
fontpath = "./simsun.ttc" # <== 这里是宋体字体路径  存在放jing_vision/utils下
font = ImageFont.truetype(fontpath, 32)
img= cv2.imread('xinan_baidu.png')
img = np.maximum(img,50)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((50, 80),  "端午节就要到了。。。", font = font, fill = (100, 1, 1, 1))
img = np.array(img_pil)

cv2.putText(img,  "--- by Silencer", (200,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1,1,1), 1, cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)