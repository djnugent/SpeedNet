import cv2
import numpy as np
import json


with open('drive.json') as json_data:
    raw = json.load(json_data)
    speed_data = np.array(raw, dtype = 'float32')[:,1]
cap = cv2.VideoCapture('drive.mp4')


'''
for i in range(0,2000):
'''
ret, frame1 = cap.read()
frame1 = frame1[200:400]
frame1 = cv2.resize(frame1, (0,0), fx=0.3, fy=0.5, interpolation = cv2.INTER_AREA)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fn = 0
while(1):
    ret, frame2 = cap.read()
    frame2 = frame2[200:400]

    frame2 = cv2.resize(frame2, (0,0), fx=0.3, fy=0.5, interpolation = cv2.INTER_AREA)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, 0.4, 1, 12, 2, 8, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = (mag * 15).astype(int)#cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2a',bgr)
    bgr = cv2.resize(bgr, (0,0), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
    bgr = cv2.resize(bgr, (0,0), fx=5, fy=5, interpolation = cv2.INTER_AREA)
    cv2.imshow('frame2b',bgr)
    print speed_data[fn]
    fn += 1
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    prvs = nxt
cap.release()
cv2.destroyAllWindows()
