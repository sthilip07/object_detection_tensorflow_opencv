import flask
import requests
from flask import Flask, request, jsonify
import warnings
import json
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from urllib.request import urlopen
import urllib.request
import logging
import sys

app = Flask(__name__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

warnings.filterwarnings('ignore')


@app.route("/predict", methods=['POST'])
def predict():
    #data = {}  # dictionary to store result
    #data["success"] = False
    if request.is_json:
        js = request.get_json()
    url_name = js['publicUrl']
    
    config_file = 'saved_model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'saved_model/frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classlables = []
    file_name = 'saved_model/labels.txt'
    with open(file_name, 'rt') as fpt:
        classlables = fpt.read().rstrip('\n').split('\n')

    # print(len(classlables))

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    # import PIL.Image
    # rgba_image = PIL.Image.open('sample.jpg')
    # rgb_image = rgba_image.convert('RGB')
    # rgb = cv2.cvtColor("sample.jpg", cv2.COLOR_RGBA2RGB)

    urllib.request.urlretrieve(url_name, "sample.jpg")
    img = cv2.imread('sample.jpg')
    classindex, confidence, bbox = model.detect(img, confThreshold=0.5)
    lst = []
    for i in classindex:
        if i<12:
            lst.append(classlables[i - 1])
        else:
            lst.append(classlables[i - 2])
    d = {x:lst.count(x) for x in lst}

    # conf = []
    # for i in confidence:
    #     conf.append(float(i))

    # dict = {"file_name":url_name,"objects":lst,"confidence":conf}
    dict = {"publicUrl":url_name,"objects":d}

    return(jsonify(dict))

    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8928,debug=True)
