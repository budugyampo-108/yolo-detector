# Image Detector (YOLO)

A web app that detects objects in an uploaded image using three YOLO11n
models trained on my own annotated photos (Ultralytics Platform).

## How it works
- Upload an image
- Each model returns the detected class and a confidence percentage
- If nothing is detected, the app reports no match

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Live app
https://yolo-detector-nzc5uxpdfnkw5ncdjledzf.streamlit.app

## Author
Francis Budu Gyampo, Pentecost University 
