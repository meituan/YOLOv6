#!/usr/bin/env python3

from tools.infer import run
import streamlit as st
from PIL import Image
import numpy as np
import os

def home():
    st.image(Image.open(os.path.join('assets', 'logo.png')), use_column_width=False, width=500)
    st.write('# About YOLO')
    st.write('YOLO (You Only Look Once) is an open source project focusing on the development of an end-to-end neural network making multiple predictions over a single iteration. MT-YOLOv6 was inspired by the original one-stage YOLO architecture and thus was (bravely) named YOLOv6 by its authors! This web application provides a platform to play around with YOLOv6. Enjoy, and read more [here](https://dagshub.com/blog/yolov6/)!')
    inference()
    
def inference():
    st.sidebar.header('Mode Selection')
    menu = ['Upload an Image', 'Take a Picture']
    choice = st.sidebar.selectbox('Select an option:', menu)
    
    file = None
    if choice == 'Take a Picture':
        st.write('### Take a picture')
        file = st.camera_input("Say cheese!")
    else:
        st.write('### Upload Image')
        file = st.file_uploader('Select an image from your local files:', type=['jpg', 'jpeg', 'png'])

    if file is not None:
        file = np.array(Image.open(file).convert('RGB'))
        res = run(weights=os.path.join('yolov6', 'weights', 'yolov6s.pt'),
                source=file,
                yaml=os.path.join('data', 'coco.yaml'))
        if res is not None:
            st.image(res, caption='YOLOv6 Inference')
        else:
            st.write('Invalid Image!')

def main():
    st.set_page_config(page_title='YOLOv6 Playground', page_icon=os.path.join('assets', 'favicon.ico'), layout='wide')

    st.title('Welcome to the (unofficial) YOLOv6 Playground!')
    home()

if __name__ == '__main__':
    main()
