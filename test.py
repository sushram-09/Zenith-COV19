from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import math
from modules.detection import detect_people,detect_boundary
from scipy.spatial import distance as dist
from modules.config import camera_no
import matplotlib.pyplot as plt
from modules.config import MIN_CONF,confidence_threshold

#Labels defined
labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

#Colours defined
COLORS = np.random.randint(0,
                        255,
                        size=(len(LABELS), 3),
                        dtype="uint8")

#YOLO algo imported
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Load pretrained face ded model
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#Trained Mask model
model_store_dir= "models/classifier.model" 
maskNet = load_model(model_store_dir)

cap = cv2.VideoCapture("mask2.mp4")       #Start Video Capturing

while(cap.isOpened()):

    #Reading the image
    flag,frame = cap.read()
    if (flag == False):
        break

    frame = cv2.resize(frame, (720, 640))
    (H,W) = frame.shape[:2]
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #Output layer names


    #* PREDICTING PEOPLE
    #Processing the frame
    blob = cv2.dnn.blobFromImage(frame,
                                1/255.0,
                                (416, 416),
                                swapRB=True,
                                crop=False)

    results,boxes,idxs = detect_boundary(frame, net, ln, MIN_CONF, personIdx=LABELS.index("person"))
    
    
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Time taken to predict the image: {:.6f}seconds".format(end-start))

    #* CHECK DISTANCE
    a = []
    b = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
    
    distance = []
    not_safe = []
    safe = []
    for i in range(0, len(a) - 1): # 0 1
        for k in range(1, len(a)):
            if (k == i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)

                if (d <= 200.0):
                    not_safe.append(i)
                    not_safe.append(k)
                else:
                    safe.append(i)
                    safe.append(k)

                safe = list(set(safe).difference(not_safe))
                not_safe = list(dict.fromkeys(not_safe))
                safe = list(dict.fromkeys(safe))

    print("not safe=",not_safe)
    print("safe=",safe)


    color = (0, 0, 255)
    for i in not_safe:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        text = "NOT SAFE"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    print(idxs)

    color = (138, 68, 38)
    for i in safe:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        text = "SAFE"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    #* FACE DETECTION
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (416, 416), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence =detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            #* MASK DETECTION
            (mask, without_mask) = maskNet.predict(face)[0]
            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
            c = cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            x = cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
            print("End of classifier")
    
    cv2.imshow("Image",frame)
    if cv2.waitKey(75) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()   
