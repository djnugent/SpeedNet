from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np
import cv2
import sys
import os
import argparse
import json
import shutil


class SpeedNet:

    DSIZE = (100,100)
    W_FILE = "weights.h5"
    EPOCHS = 50
    BATCH_SIZE = 32

    def main(self, args):
        #compile model
        self.create_model()

        self.optflow_dir = args.video_file.split('.')[0] + "_optflow"

        #train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                self.load_weights()
            #start training session
            self.train(args.video_file,args.speed_file,args.split,args.wipe,self.EPOCHS,self.BATCH_SIZE)

        #test the model
        elif args.mode == "test":
            self.test(args.video_file,args.speed_file)

        elif args.mode == "play":
            self.play(args.video_file,args.speed_file)


    def process_frame(self,frame):
        frame = cv2.resize(frame, self.DSIZE, interpolation = cv2.INTER_AREA)
        frame = frame/127.5 - 1.0
        return frame

    def optflow(self,frame1,frame2):
        frame1 = frame1[200:400]
        frame1 = cv2.resize(frame1, (0,0), fx = 0.4, fy=0.5)
        frame2 = frame2[200:400]
        frame2 = cv2.resize(frame2, (0,0), fx = 0.4, fy=0.5)
        flow = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow_data = cv2.calcOpticalFlowFarneback(prev, nxt, 0.4, 1, 12, 2, 8, 1.2, 0)
        #convert data to hsv
        mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
        flow[...,1] = 255
        flow[...,0] = ang*180/np.pi/2
        flow[...,2] = (mag *15).astype(int)
        return flow

    def prep_data(self,video_file,json_file, shuffle = False,wipe = False):
        print "Prepping data"
        #decode json speed data
        print "Decoding json"
        f = open(json_file,'r')
        data = json.load(f)
        speed_data = np.array(data[0:-1], dtype = 'float32')[:,1]
        print "loaded " + str(len(speed_data)) + " json entries"

        #clear preprocessed data
        if wipe and os.path.isdir(self.optflow_dir):
            print "wiping preprocessed data..."
            shutil.rmtree(self.optflow_dir)

        #process video data if it doesn't exist
        processed_video = None
        if not os.path.isdir(self.optflow_dir):
            print "preprocessing data..."
            os.mkdir(self.optflow_dir)
            #Decode video frames
            vid = cv2.VideoCapture(video_file)
            frame_cnt = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            processed_video = np.empty((frame_cnt-1,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8')
            ret, prev = vid.read()
            i = 0
            while True:
                ret, nxt  = vid.read()
                if not ret: #EOF
                    break
                #crop and resize frame
                flow = self.optflow(prev,nxt)
                prev = nxt
                flow = cv2.resize(flow, self.DSIZE, interpolation = cv2.INTER_AREA)
                processed_video[i] = flow/127.5 - 1.0
                cv2.imwrite(self.optflow_dir + '/' + str(i) + ".png", flow)
                sys.stdout.write("\rProcessed " + str(i) + " frames" )
                i+=1
            print "\ndone processing " + str(frame_cnt) + "frames"
        #preprocessed data exists
        else:
            print "Found preprocessed data"
            frame_cnt = len(os.listdir(self.optflow_dir))
            processed_video = np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='float32')
            for i in range(0,frame_cnt):
                flow = cv2.imread(self.optflow_dir + '/' + str(i) + ".png")
                processed_video[i] = flow/127.5 - 1.0
                sys.stdout.write("\rLoading frame " + str(i))
            print "\ndone loading " + str(frame_cnt) + " frames"

        #shuffle data
        if(shuffle):
            print "Shuffling data"
            randomize = np.arange(len(processed_video))
            np.random.shuffle(randomize)
            processed_video = processed_video[randomize]
            speed_data = speed_data[randomize]

        print "Done prepping data"
        return (processed_video, speed_data)


    def create_model(self):

        print "Compiling Model"

        self.model = Sequential()
        self.model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),input_shape=(self.DSIZE[0],self.DSIZE[1],2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')


    def load_weights(self):
        try:
            print "loading weights"
            self.model.load_weights(self.W_FILE)
            return True
        except ValueError:
            print "Unable to load weights. Model has changed"
            print "Please retrain model"
            return False
        except IOError:
            print "Unable to load weights. No previous weights found"
            print "Please train model"
            return False

    def train(self,X_src,Y_src,val_split, wipe,n_epochs = 50, batch_size= 32):
        #load data
        X,Y = self.prep_data(X_src,Y_src,shuffle = True,wipe = wipe)
        X = X[:,:,:,[0,2]] #extract channels with data
        #train model
        print "Starting training"
        self.model.fit(X, Y, batch_size=batch_size,
                    nb_epoch=n_epochs,validation_split=val_split)
        #save weights
        print "Done training. Saving weights"
        self.model.save_weights(self.W_FILE)

    def test(self,X_src, Y_src):
        #load data
        X_test,Y_test = self.prep_data(X_src,Y_src,shuffle = False)
        X_test = X_test[:,:,:,[0,2]] #extract channels with data

        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            print "Starting testing"
            print self.model.evaluate(X_test,Y_test)
            print "Done testing"
        else:
            print "Test failed to complete with improper weights"

    def play(self, X_src, Y_src):
        print "Starting testing"
        #load data
        X,Y = self.prep_data(X_src,Y_src,shuffle = False)
        rec = cv2.VideoWriter('flow.avi',int(cv2.cv.CV_FOURCC('M','J','P','G')),48,(300,300))
        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            for x,y in zip(X,Y):
                flow = ((x + 1) * 127.5).astype('uint8')
                flow = cv2.cvtColor(flow,cv2.COLOR_HSV2BGR)
                flow = cv2.resize(flow,(300,300))
                pred_y = self.model.predict(np.array([x[:,:,[0,2]]]))[0,0]
                error = abs(y-pred_y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(flow,"Predicted Speed: " + str(pred_y),(5,15),font, 0.55,(255,255,255),2)
                cv2.putText(flow,"Actual Speed: " + str(y),(5,45),font, 0.55,(255,255,255),2)
                cv2.putText(flow,"Error: " + str(error),(5,75),font, 0.55,(255,255,255),2)
                rec.write(flow)
            rec.release()
            print "Done predicting"
        else:
            print "Prediction failed to complete with improper weights"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file",
                        help="video file name")
    parser.add_argument("speed_file",
                        help="json speed data file name")
    parser.add_argument("--split", type=float, default=0,
                        help="percentage of train data for validation")
    parser.add_argument("--mode", choices=["train", "test", "play"], default='train',
                        help="Train, Test, or Play model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    print "Running SpeedNet by Daniel Nugent"
    net = SpeedNet()
    net.main(args)
