import json
import cv2

img = cv2.imread('map.png')
with open('temp.json','r') as f:
    map_points = json.load(f)

for point in map_points:
    for point in map_points:
        # print(type(point),point)
        # print(point[0], point[1])
        cv2.circle(img,(point[0],point[1]),4,(125,125,125),-1)

cv2.imshow('img',img)
cv2.waitKey(1)
