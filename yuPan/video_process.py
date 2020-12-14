"""
从摄像头保存视频
"""
import cv2
import time

class SaveVideo:
    def __init__(self,src=None):
        """
        src 为录制视频路径
        例如：0 表示电脑默认第一个摄像头
            rtsp地址如下：rtsp://admin:xxl12345@192.168.8.69:554/MPEG-4/ch1/sub/av_stream
        """
        if src==None:
            self.src = 'rtsp://admin:xxl123456@192.168.8.69:554/MPEG-4/ch1/sub/av_stream'
        else:
            self.src = src

    def save_video(self):
        cap = cv2.VideoCapture(self.src)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')   #保存为AVI
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')   #保存为MP4
        struct_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        out = cv2.VideoWriter('./videos/%s.avi'%(struct_time), fourcc, 10.0, (704, 576))  # 图像大小参数按（宽，高）一定得与写入帧大小一致

        while (True):
            ret, frame = cap.read()
            print(frame.shape)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def show_video(self):
        cap = cv2.VideoCapture(self.src)
        while (True):
            ret, frame = cap.read()
            if frame is None:
                continue
            print(frame.shape)
            resize_frame = cv2.resize(frame,(1080,640))
            print(resize_frame.shape)
            cv2.imshow('frame', resize_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # obj = SaveVideo(src = 'rtsp://admin:lukuang123@192.168.3.133:554/MPEG-4/ch1/sub/av_stream')
    obj = SaveVideo(src = 'rtsp://admin:123456@192.168.1.100:554/ch1/0')
    # obj.save_video()
    obj.show_video()

