#!/usr/bin/env python
import cv2
import cv2.cv as cv
import numpy as np
import scipy.misc as sci
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
	return sum(l) / float(len(l))

def get_drowsiness(img, eyesize):
	#this is just horizontal symmetry for the image
	copy = cv.fromarray(img.copy())#meaning we just use cv array here
	topcopy = copy[0:(eyesize // 2), 0:(eyesize)]
	bottomcopy = copy[(eyesize // 2):(eyesize), 0:(eyesize)]
	cv.SaveImage("topcopy.jpg", topcopy)
	cv.SaveImage("bottomcopy.jpg", bottomcopy)
	diff = cv.CreateMat(eyesize // 2, eyesize, 0)#0 is the type
	cv.AbsDiff(topcopy, bottomcopy, diff)
	cv.SaveImage("diff.jpg", diff)
	score = cv.Sum(diff)[0]
	print "Score: ", score

def get_eyepic(img, eyesize):
	img_copy = img.copy()
	img_height = img_copy.shape[0]
	img_width = img_copy.shape[1]
	print "got eye"
	sub = img_copy[(img_height // 3) + (img_height // 10) : 2 * (img_height // 3), :]#the 8 is a bit of a hack
	sub = sci.imresize(sub, (eyesize, eyesize))
	get_drowsiness(sub, eyesize)
	cv2.imwrite("eye.jpg", sub)

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	cascade = cv2.CascadeClassifier("./haardetectors/haarcascade_frontalface_alt.xml")
	nested = cv2.CascadeClassifier("./haardetectors/haarcascade_eye.xml")
	counter1 = 19 #counter for taking new img
	counter2 = 0
	facenum = 0 #index for training face
	eyenum = 0 #index for training eyes

	#facerect
	x1s, y1s, x2s, y2s = 0, 0, 0, 0

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
			x1s, y1s, x2s, y2s = x1, y1, x2, y2
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
					get_eyepic(sub_roi, 20) #numerical params are size of eye
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
