# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    run_the_app()


# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():


    st.sidebar.markdown("# 检测视频")
    # 视频文件名称列表
    video_name = st.sidebar.selectbox("选择待检测视频", ['v11010916','v11021543','v11050810'])
    st.sidebar.markdown("# 模型阈值设置")
    confidence_threshold = st.sidebar.slider("置信度：", 0.0, 1.0, 0.9, 0.01)
    video_path='%s.mp4'%video_name
    pre = '还未检测'
    agree = st.checkbox('开始检测')

    video_file = open(video_path,'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    video_name_map_dict={'v11010916':'v1101091613','v11021543':'v1102154400','v11050810':'v1105081042'}
    if agree:
        pre_label, p = detect_video(video_path)
        st.subheader('标签：' + pre_label)
        st.subheader('概率：' + str(p))
        video_name_map=video_name_map_dict[video_name]
        m,d,h,mi,s=video_name_map[1:3],video_name_map[3:5],video_name_map[5:7],video_name_map[7:9],video_name_map[9:]
        end_mi = int(mi)
        end_s = int(s)+30
        if end_s>60:
            end_s = end_s%60
            end_mi=end_mi+1
            end_mi,end_s = str(end_mi),str(end_s)
        st.subheader('时间：' + 'start:%s月%s日 %s:%s:%s--end:%s月%s日 %s:%s:%s'%(m,d,h,mi,s,m,d,h,end_mi,end_s))


def process_img(frame):
    img = cv2.resize(frame,(224,224))
    img = np.asarray(img[:, :, (2, 1, 0)], dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def detect_video(video_path):
    import cv2
    import keras
    # @st.cache(allow_output_mutation=True)
    # @st.cache(allow_output_mutation=True)
    def load_model():
        t = st.subheader('loading model...')
        print('loading model...')
        model = keras.models.load_model('model.h5')
        return model,t
    model,t = load_model()
    t.empty()
    # pre = detect_video(video_path, model)
    cate_map = ['排污水', '未排污水']
    videoCapture = cv2.VideoCapture(video_path)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    w = videoCapture.get(3)
    h = videoCapture.get(4)
    print('w,h', w, h)
    print('fps:', fps)
    frame_count = 0
    while True:
        # print('当前帧:', fream_count)
        rval, frame = videoCapture.read()
        if rval is False:
            print('video is over')
            break
        frame_count += 1
        if frame_count % (int(fps) * 2) == 0:
            img = process_img(frame)
            pre = model.predict(img)
            #         print(cate,cate_map[np.argmax(pre[0])])
            label = cate_map[np.argmax(pre[0])]
            p = np.max(pre[0])
            return label,p
    videoCapture.release()


if __name__ == "__main__":
    main()
