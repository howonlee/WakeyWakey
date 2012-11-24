#!/usr/bin/env python
import cv2
import cv2.cv as cv
import os

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize = (30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def avg(l):
    return sum(l) / len(l)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier("./haardetectors/haarcascade_frontalface_alt.xml")
    nested = cv2.CascadeClassifier("./haardetectors/haarcascade_eye.xml")
    counter1 = 0 #counter for taking new img
    counter2 = 0
    facenum = 0 #index for training face
    eyenum = 0 #index for training eyes

    while True:
        ret, im = cap.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detect(gray, cascade)
        vis = im.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            if (counter1 > 20):
                counter1 = 0
                facenum += 1
                print "saving images"
                cv2.imwrite('trainface' + str(facenum) + '.jpg', roi)
                eyepos = []
                for x3, y3, x4, y4 in subrects:
                    eyenum += 1
                    sub_roi = roi[y3:y4, x3:x4]
                    pos = [avg([x3, x4]), avg([y3, y4])]
                    eyepos.append(pos)
                    cv2.imwrite('traineye' + str(eyenum) + '.jpg', sub_roi)
            if (counter2 > 200):
                counter2 = 0
                #now train this crap
                print "training on images"
            draw_rects(vis_roi, subrects, (255, 0, 0))
            counter1 += 1
            counter2 += 1
        cv2.imshow('video test', vis)
        key = cv2.waitKey(10)
        if key == 27:
            break
        if key == ord(' '):
            cv2.imwrite(picdir + 'vid_result.jpg', vis)
