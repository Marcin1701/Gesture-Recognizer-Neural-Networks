import tflearn
import numpy as np
import cv2
import imutils
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
from PIL import Image

bg = None


def getPredictedGestureName(predictedClass):
    return {
        0: "Swing",
        1: "Palm",
        2: "Fist",
        3: "None",
        4: "Ok",
        5: "Peace",
        6: "Straight",
        7: "Thumb"
    }.get(predictedClass, "")


def getConfidenceFontColors(confidence):
    if confidence < 25:
        return 0, 0, 255
    elif 25 <= confidence < 75:
        return 0, 239, 255
    elif confidence >= 75:
        return 0, 255, 0


def resizeImage(imageName):
    baseWidth = 100
    img = Image.open(imageName)
    wpercent = (baseWidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((baseWidth, hsize), Image.ANTIALIAS)
    img.save(imageName)


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented


def main():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    start_recording = False
    while True:
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (_, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        start_recording = True
        if keypress == ord("q"):
            break


def getPredictedClass():
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (
        prediction[0][0] + prediction[0][1] + prediction[0][2] +
        prediction[0][3] + prediction[0][4] + prediction[0][5] +
        prediction[0][6] + prediction[0][7]))


def showStatistics(predictedClass, confidence):
    textImage = np.zeros((300, 512, 3), np.uint8)
    className = getPredictedGestureName(predictedClass)
    cv2.putText(textImage, "Gesture: " + className, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(textImage, 
                "Confidence: " + str(confidence * 100) + '%', 
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                getConfidenceFontColors(confidence * 100),
                2)
    cv2.imshow("Statistics", textImage)


ops.reset_default_graph()
net = input_data(shape=[None, 89, 100, 1], name='input')
net = conv_2d(net, 32, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 256, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 256, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 2, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 2, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 1000, activation='relu')
net = dropout(net, 0.75)
net = fully_connected(net, 8, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='regression')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load("TrainedModel/GestureRecogModel.tfl")

if __name__ == "__main__":
    main()
