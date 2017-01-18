import numpy as np
import cv2
import math
import json

f = open('drive.json')
data = json.load(f)
speed = np.array(data)[:,1]

cap = cv2.VideoCapture('drive.mp4')
#params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 2,
                       blockSize = 10 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(400,3))



def optflow(old, new):
    old_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    tot = 0.0
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        tot += math.sqrt(pow(a-c,2) + pow(b-d,2))
        #cv2.circle(new,(a,b),5,color[i].tolist(),-1)
    mag = tot/i
    return mag


# Take first frame and find corners in it
#for i in range(0,3000):
#    cap.grab()

ret, frame = cap.read()
LFrame_old = frame[200:400,0:250]
RFrame_old = frame[200:400,300:600]

# Create a mask image for drawing purposes
i = 0
while(1):
     ret,frame = cap.read()
     LFrame = frame[200:400,0:250]
     RFrame = frame[200:400,300:600]
     Lmag = optflow(LFrame_old,LFrame)
     Rmag = optflow(RFrame_old,RFrame)
     print int(Rmag), int(Lmag), speed[i]

     cv2.imshow('Lframe',LFrame)
     cv2.imshow('Rframe',RFrame)
     k = cv2.waitKey(15) & 0xff
     if k == 27:
         break

     # Now update the previous frame and previous points
     LFrame_old = LFrame
     RFrame_old = RFrame
     i+=1

cv2.destroyAllWindows()
cap.release()
