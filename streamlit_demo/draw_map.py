import cv2
import numpy as np

drawing = False # 鼠标左键按下时，该值为True，标记正在绘画
mode =False  # True 画矩形，False 画圆
ix, iy = -1, -1 # 鼠标左键按下时的坐标

return_xy=[]
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # 鼠标移动事件
        if drawing == True:
            if mode == True:
                print(ix, iy)
                print(x, y)
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                print(x,y)
                return_xy.append([x,y])
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键松开事件
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


# img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread('draw.png')
cv2.namedWindow('image')
print(img.shape)
cv2.setMouseCallback('image', draw_circle) # 设置鼠标事件的回调函数
try:
    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
except KeyboardInterrupt:
    print(return_xy)
    import json
    with open('temp.json','w') as f:
        json.dump(return_xy,f)
cv2.destroyAllWindows()