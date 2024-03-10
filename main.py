import os
from ultralytics import YOLO
import cv2
import numpy as np
import time
import copy
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

path = './'
window_size = [1080, 720]

# Detector 관련
detector = YOLO(f'{path}/yolov8n-face.pt')
class_detector = {0:'얼굴'}

# Classifier 관련
classifier = YOLO(f'{path}/classifier.pt') # 8가지 표정
class_classifier = {0:'무표정', 1:'행복', 2:'슬픔', 3:'놀람', 4:'두려움', 5:'혐오', 6:'화남', 7:'경멸'}

def run_classifier(classifier, class_classifier, input_img):
    results = classifier.predict(source = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), verbose = False)[0]
    # top5 도출
    prob_list = results.probs.data.tolist()
    top5 = {}
    for class_name, prob in zip(list(results.names.values()), prob_list):
        top5[class_classifier[int(class_name)]] = round(prob, 1)
    top5 = {k: v for k, v in sorted(top5.items(), key=lambda item: item[1], reverse=True)}
    # class_name, conf 도출
    for key, value in top5.items():
        class_name = key
        conf = value
        break
    return {'top5':top5, 'class_name':class_name, 'conf':conf}

def run_detector(input_img):
    dic_list = []
    # Detector 인퍼런스
    h, w, c = input_img.shape
    results = detector.predict(source = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), verbose = False, conf = 0.3)[0]
    results = results.boxes.data.tolist()
    for result in results:
        x1, y1, x2, y2, conf, cls = int(result[0]), int(result[1]), int(result[2]), int(result[3]), float(result[4]), int(result[5])
        # 리사이즈 bbox 계산
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dic = {'bbox':[x1, y1, x2, y2], 'conf':round(conf, 3), 'cls':cls, 'class_name': class_detector[cls], 'img_size':[w, h], 'mode':'1stage'}
        # Classifier 구동
        crop_img = input_img[y1:y2, x1:x2]
        result_classifier = run_classifier(classifier, class_classifier, crop_img)
        dic['class_name'] = result_classifier['class_name']
        dic['conf'] = result_classifier['conf']
        dic['top5'] = result_classifier['top5']
        dic_list.append(dic)
    return dic_list

# 결과 그리기
def draw_img(img, dic_list):
    h, w, c = img.shape
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for dic in dic_list:
        bbox_letter_x, bbox_letter_y = dic['bbox'][0]+3, dic['bbox'][1]
        draw.rectangle([dic['bbox'][0], dic['bbox'][1], dic['bbox'][2], dic['bbox'][3]], outline=(0,0,255), width=1)
        draw.text((bbox_letter_x, bbox_letter_y), f'{dic["class_name"]}:{dic["conf"]}', (0,0,255), ImageFont.truetype('NanumGothic.ttf', 18))
    img = np.array(img_pil)
    return img

# 인퍼런스
cap = cv2.VideoCapture(0)
while True:
    # 이미지 수신
    ret, img = cap.read()
    if ret == False:
        print('카메라 미수신... 재시도')
        time.sleep(0.3)
        continue
    # 인퍼런스
    dic_list = run_detector(img)
    img = draw_img(img, dic_list)
    img = cv2.resize(img, (window_size[0], window_size[1]))
    cv2.imshow('window', img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
