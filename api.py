import json
import re

from PIL import Image
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing  import image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

model = tf.keras.models.load_model("train1.h5")
per=1
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)

def image_processing_CCCD_Chip(img):
    orb = cv2.ORB_create(100000)  # nếu cái ảnh form mà nền nhiều màu khó phân biệt thì tăng 100000 cái form 1000
    imgQ = cv2.imread('CCCDChip_Form.png')
    h, w, c = imgQ.shape
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    impKp1 = cv2.drawKeypoints(imgQ, kp1, None)
    # img = cv2.imread(filename)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    crop_image = imgScan[0:275, 0:466]
    # path1 = 'Crop2'
    # cv2.imwrite(path1 +"/"+ "cropimage.png",crop_image)
    # # cv2.imshow(filename, crop_image)
    # path = path1 +"/"+ "cropimage.png"  # path anh ne, anh cat roi ay
    crop_image = Image.fromarray(crop_image)
    tree = ET.parse('CCCDChip_Form.xml')  # path xml
    root = tree.getroot()
    wForm = int(root.find('size')[0].text)
    hForm = int(root.find('size')[1].text)
    area = []
    for box in root.iter('object'):
        area.append([box[0].text, int(box[4][0].text), int(box[4][1].text), int(box[4][2].text), int(box[4][3].text)])

    crop_image = crop_image.resize((wForm, hForm))

    data = []
    for box in area:
        label = box[0]
        crop111 = crop_image.crop(box[1:5])
        a = detector.predict(crop111, True)
        if (a[1] > 0.75):
            data.append(a[0])
        else:
            data.append("")

    # print(data)
    info = {
        "ID": data[0].strip().strip("-"),
        "Name": data[1].strip(),
        "Birthday": data[2].strip(),
        "Gender": data[3].replace('-','').strip(),
        "Nationality": data[4].strip(),
        "Home": data[5].strip(),
        "Address": data[6].strip()+", "+data[7].strip()

    }
    # os.remove(path)
    for key in info:
        print(key, ' : ', info[key])
    return info


def image_processing_CCCD(img):
    orb = cv2.ORB_create(100000)  # nếu cái ảnh form mà nền nhiều màu khó phân biệt thì tăng 100000 cái form 1000
    imgQ2 = cv2.imread('CCCD_Form.png')          #form
    h1, w1, c1 = imgQ2.shape
    kp12, des12 = orb.detectAndCompute(imgQ2, None)
    impKp12 = cv2.drawKeypoints(imgQ2, kp12, None)

    # img = cv2.imread(filename)
    kp22, des22 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des22, des12)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp22, imgQ2, kp12, good[:100], None, flags=2)

    srcPoints = np.float32([kp22[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp12[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w1, h1))
    crop_image2 = imgScan[0:240, 0:388]
    # path1 = 'Crop2'
    # cv2.imwrite(path1 + "/" + "cropimage2.png", crop_image2)
    # # cv2.imshow(filename, crop_image)
    # path = path1 + "/" + "cropimage2.png"  # path anh ne, anh cat roi ay
    crop_image = Image.fromarray(crop_image2)
    tree = ET.parse('CCCD_Form.xml')  # path xml
    root = tree.getroot()
    wForm = int(root.find('size')[0].text)
    hForm = int(root.find('size')[1].text)
    area = []
    for box in root.iter('object'):
        area.append([box[0].text, int(box[4][0].text), int(box[4][1].text), int(box[4][2].text), int(box[4][3].text)])

    crop_image = crop_image.resize((wForm, hForm))

    data = []
    for box in area:
        label = box[0]
        crop111 = crop_image.crop(box[1:5])
        a = detector.predict(crop111,True)
        print(a)
        if (a[1] > 0.75):
            data.append(a[0])
        else:
            data.append("")


    # print(data)
    info = {
        "ID": data[0].strip(),
        "Name": data[1].strip(),
        "Birthday": data[2].strip(),
        "Gender": data[3].strip(),
        "Nationality": data[4].strip(),
        "Home": (data[5].strip()+', '+data[6].strip()).strip(', '),
        "Address": (data[7].strip()+', '+data[8].strip()).strip(', ')
    }

    # os.remove(path)
    for key in info:
        print(key, ' : ', info[key])
    return info

def image_processing_CMT(img):
    orb = cv2.ORB_create(100000)  # nếu cái ảnh form mà nền nhiều màu khó phân biệt thì tăng 100000 cái form 1000
    imgQ1 = cv2.imread('CMT_Form.jpg')
    h12, w12, c12 = imgQ1.shape
    kp13, des13 = orb.detectAndCompute(imgQ1, None)
    impKp13 = cv2.drawKeypoints(imgQ1, kp13, None)

    # img = cv2.imread(filename)
    kp23, des23 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des23, des13)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp23, imgQ1, kp13, good[:100], None, flags=2)

    srcPoints = np.float32([kp23[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp13[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w12, h12))
    crop_image3 = imgScan[0:250, 0:388]

    # path1 = 'Crop2'
    # cv2.imwrite(path1 + "/" + "cropimage3.png", crop_image3)
    # # cv2.imshow(filename, crop_image)
    # path = path1 + "/" + "cropimage3.png"  # path anh ne, anh cat roi ay
    crop_image = Image.fromarray(crop_image3)
    tree = ET.parse('CMT_Form.xml')  # path xml
    root = tree.getroot()
    wForm = int(root.find('size')[0].text)
    hForm = int(root.find('size')[1].text)
    area = []
    for box in root.iter('object'):
        area.append([box[0].text, int(box[4][0].text), int(box[4][1].text), int(box[4][2].text), int(box[4][3].text)])
    crop_image = crop_image.resize((wForm, hForm))

    data = []
    for box in area:
        label = box[0]
        crop111 = crop_image.crop(box[1:5])
        a = detector.predict(crop111, True)
        print(a)
        if(a[1]>0.75):
            data.append(a[0])
        else: data.append("")


    # print(data)
    info = {
        "ID": data[0],
        "Name": (data[1]+' '+data[2]).strip(),
        "Birthday": data[3],
        "Home": (data[4]+', '+data[5]).strip(','),
        "Address": (data[6]+', '+data[7]).strip(',')
    }
    # os.remove(path)
    for key in info:
        print(key, ' : ', info[key])
    return info

app = Flask(__name__)
@app.route("/upload", methods=['POST'])
def classification():
    if request.method == 'POST':

        f = request.files['file']
        f.save(secure_filename('img_request.png'))
        classNames = {0: 'CCCD', 1: 'CCCDChip', 2: 'CMT'}
        img = image.load_img('img_request.png', target_size=(150, 150))
        x = image.img_to_array(img) / 255
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        print(result)
        result = np.argmax(result)

        img = cv2.imread('img_request.png')
        data={}
        if (result == 2):
            print("CMT")

            data = image_processing_CMT(img)



        elif (result == 0):
            print("CCCD")
            data = image_processing_CCCD(img)

        else:
            print("CCCDChip")
            data = image_processing_CCCD_Chip(img)


        os.remove('img_request.png')

        return json.dumps(data, ensure_ascii=False)


@app.route('/')
def home():
    return render_template('index.html')
if __name__ == "__main__":
    app.run()
