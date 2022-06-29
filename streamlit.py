#!/usr/bin/env python3

from tools.infer import run
import streamlit as st
from PIL import Image
import numpy as np
import os

def home():
    st.markdown('[![dagshub](https://raw.githubusercontent.com/nirbarazida/YOLOv6/streamlit/assets/logo.png)](https://dagshub.com/)', unsafe_allow_html=True)
    st.markdown('<a href="https://dagshub.com/user/sign_up?redirect_to=" title="DAGsHub Sign Up"><img src="https://img.shields.io/badge/DagsHub-Sign%20Up-%231F4C55?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAACXBIWXMAAACXAAAAlwHUBiyCAAACmUlEQVRIieVXsYoUQRB9nouY7QQGBoITGumYiCDimBiYuKZnMibG+wfOJ6z+wE2kmbbBgSa6CwoqCHtofnuIoTCbiUlJ6euz7O3p23U9LvBBM71d3fVqqqqrZo+JCI4CG0fCCqDHZwZgyPmY41CxYciUvAXgAIwOgXRAnp+6lTgnqSNxzuGN+ReoSPiIBhQQkVJExiJSyy9MRaQQkUZEZiJSiUimSfgXQ8+NqPOOiLTkKr2wJYFFRaMayusViD1hy/NXOW9pRN6je9Wt5wKXbgG4SzflDIWiBlBG3O8TMuN8ynMt530AEwDXAMx8VjckCLFl5BqbXQD3qWBGpRkJChJkfHp9augFzr/wLKxrvBtiKLhvEMTbmb0au3t0sZcXga6HXm7josTbHcRNRyzrYF8oD19km8b/Ubk0hvOIuyemuKSwE5HZevAJwE2fC5a4oeCNWZsztu0SxLE9qvMB5+8APPP7LPGYG/u0DkyQZUhT0OT6DOCG9UDYJNSlHwAcB/DYXKF1oIY/AfDS9oBYd1Ky9wA2I7L8gN9dcOHZGHHBOxpbnzIcdi00KrzHiIVrlX5cMf4D00h07HH4hlKyYNibsEDcCxcS8IoaEtY0pG+OOK7fDnr6QkhWIQZdOGK99djhG6nbb3FMTA+A6cW/Eak2VaRSlWyRqYqWR+RaRs+zKuZWZ4w4J4kl7YIqHLImu449H0VkTj1JYpC4NPN14Tj2ObqyesQk0bidPSDurzhSyBc+oxJfEWNa2Sbe9rmZP03sdWHepD7oM1O7FRc518bxFsAJANeDM68BfAVwCsAVZvd3AJfCzE5dp5abh6Z4gM/TdN8LAN+4fhLAZSo/w7WCv0texX2s8xcmM1UMLLNu2W72n/13AvADffO77cDRF5EAAAAASUVORK5CYII="></a> | <a href="https://discord.gg/pk22NradY4" title="DagsHub on Discord"><img src="https://img.shields.io/discord/698874030052212737?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAB4AAAAXCAYAAAAcP%2F9qAAAACXBIWXMAAAsTAAALEwEAmpwYAAAFN2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNi4wLWMwMDMgNzkuMTY0NTI3LCAyMDIwLzEwLzE1LTE3OjQ4OjMyICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHRpZmY6T3JpZW50YXRpb249IjEiIHhtcDpDcmVhdGVEYXRlPSIyMDIxLTEwLTIzVDE2OjI5OjAyKzAzOjAwIiB4bXA6TW9kaWZ5RGF0ZT0iMjAyMS0xMC0yM1QxNjozNDoxMiswMzowMCIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAyMS0xMC0yM1QxNjozNDoxMiswMzowMCIgZGM6Zm9ybWF0PSJpbWFnZS9wbmciIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiIHBob3Rvc2hvcDpJQ0NQcm9maWxlPSJzUkdCIElFQzYxOTY2LTIuMSIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDpiMTBhMTRjOC1iNzg5LTQ2OTgtYmVhMi1kZTI4NDg3ZmEyMjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6YjEwYTE0YzgtYjc4OS00Njk4LWJlYTItZGUyODQ4N2ZhMjIyIiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6YjEwYTE0YzgtYjc4OS00Njk4LWJlYTItZGUyODQ4N2ZhMjIyIj4gPHhtcE1NOkhpc3Rvcnk%2BIDxyZGY6U2VxPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0ic2F2ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6YjEwYTE0YzgtYjc4OS00Njk4LWJlYTItZGUyODQ4N2ZhMjIyIiBzdEV2dDp3aGVuPSIyMDIxLTEwLTIzVDE2OjM0OjEyKzAzOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjIuMSAoTWFjaW50b3NoKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE%2BIDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24%2BIDwvcmRmOlJERj4gPC94OnhtcG1ldGE%2BIDw%2FeHBhY2tldCBlbmQ9InIiPz4jeahYAAACnElEQVRIx62WPWxOURjH7%2Fu%2BrTdN1RsGIkgw2BBLQwxCmvgamzAxCAs2xCYGSZWBWYihSUcDFhJpJJ1M0kUqJS1BlRCk%2Bno%2F7s%2FyPPV38lwtcZKTc%2B89z33%2B%2F%2BfznAzIkllO3lcD%2FcAloBbI%2B9wMDAB7ArlSKl8EWgWOAA%2BAWX6Ni8AaoA84AOwDdhm5%2ByL3ARgG9haBK2jF1rXAJL%2BPps2ikdvaANrJ3sMIXIE7bL1jP3w3sDxRjilv2WwX7DeAur2fM92dKbCD7hbmRZbli%2Fjmo2XrF2ClhtPN99g%2BWgD4X8YPWwfUao3tDnFT%2Fh%2BBPRQfJdtL6vebCcM0uRoLEGrbv61gzz14wq32GPcA7xOGHr92Qdz%2B9K0VEMdCmQFlB96fCKRZOgIMAd8Cxf48CdwCxgID%2FHnWynU%2Bq68GSeUKz0rJbQTeiTInOqqlYiTVkFz09SvwSCLo65go67b1tJB0r2yXkGXAKusDaq0bddmBu4BXiZUuNCx1Xklq3cl9NSCtkArwrEDnPQdeF7BzpU%2Bk6N3i44HFWxOLVxihSOdTB95SkFDO9Ki4uwZMyL4ru5scNteDZHV9U0A1s%2FhErU%2Bfh%2BxYnAqyui3euWAnGkHNu9wMUNOOFfXcZtA%2BF1PH9aD%2BtYMtz4BeAckDQeS0qssR2RJ3N4E5yRUKTiyAaaAnAzbI5lzSrW6biz%2F%2FRW9%2BCZwBHgtg23QDPNc67pOSQs7RF8ApYBtwGLhhzWICeAu8AcatFV6zW0mvkf1kRtRF7yiwyYH9SOwCrhTcNA4mWdtppbNUatfnYBD3GeCkXrGy9GYArLdymE5KpWqNpBRc9ErAElu7gdf27zhw3gi6XEd69SknBJYBx4yIesYvDj5LQsat3wkcCrw0T%2FonycbE%2FgQEhDUAAAAASUVORK5CYII%3D"></a> | <a href="https://www.youtube.com/c/DagsHub" title="DagsHub on Youtube"><img src="https://img.shields.io/youtube/channel/subscribers/UCeuZrCdpIY69XNWqn9OeSYQ?style=social"></a> | <a href="https://twitter.com/TheRealDAGsHub" title="DAGsHub on Twitter"><img src="https://img.shields.io/twitter/follow/TheRealDAGsHub.svg?style=social"></a>', unsafe_allow_html=True)

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
