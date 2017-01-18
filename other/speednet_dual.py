from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda, Merge
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
import numpy as np
import cv2
import sys
import os
import argparse
import json
import shutil



class SpeedNet:

    DSIZE = (80,80)
    W_FILE = "weights2.h5"
    EPOCHS = 60
    BATCH_SIZE = 32

    def main(self, args):
        #compile model
        self.create_model()

        self.optflow_dir = args.video_file.split('.')[0] + "_optflow"


        #train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                try:
                    print "loading weights"
                    self.model.load_weights(self.W_FILE)
                except ValueError:
                    print "Unable to load weights. Model has changed"
                except IOError:
                    print "Unable to load weights. No previous weights found"

            #start training session
            self.train(args.video_file,args.speed_file,args.split,args.wipe,self.EPOCHS,self.BATCH_SIZE)

        #test the model
        elif args.mode == "test":
            self.test(args.video_file,args.speed_file)

        elif args.mode == "play":
            self.play(args.video_file,args.speed_file)


    def process_frame(self,frame):
        #crop frame and reduce size
        #frame = frame[200:400]
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
        #convert data to rgb
        mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
        flow[...,1] = 255
        flow[...,0] = ang*180/np.pi/2
        flow[...,2] = (mag *15).astype(int)
        flow = cv2.cvtColor(flow,cv2.COLOR_HSV2BGR)
        return flow


    def prep_data(self,video_file,json_file, shuffle = False):
        print "Prepping data"
        #decode json speed data
        print "Decoding json"
        f = open(json_file,'r')
        data = json.load(f)
        speed_data = np.array(data, dtype = 'float32')[:,1]

        #plt.hist(speed_data *10 , bins=range(0,100))
        #plt.savefig("hist.png")

        #Decode video frames
        vid = cv2.VideoCapture(video_file)
        frame_cnt = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        processed_video = np.empty((frame_cnt,self.DSIZE[0],self.DSIZE[1],3),dtype='uint8')
        i = 0
        while True:
            ret, frame  = vid.read()
            if not ret: #EOF
                break
            #crop and resize frame
            processed_video[i] = self.process_frame(frame)
            sys.stdout.write("\rProcessed " + str(i) + " frames" )
            i+=1
        print "done processing " + str(frame_cnt) + "frames"


        #shuffle data
        if(shuffle):
            print "Shuffling data"
            randomize = np.arange(len(processed_video))
            np.random.shuffle(randomize)
            processed_video = processed_video[randomize]
            speed_data = speed_data[randomize]


        print "Done prepping data"
        return (processed_video, speed_data)


    def prep_data_new(self,video_file,json_file, shuffle = False,wipe = False):
        print "Prepping data"
        #decode json speed data
        print "Decoding json"
        f = open(json_file,'r')
        data = json.load(f)
        speed_data = np.array(data[0:-1], dtype = 'float32')[:,1]
        print "loaded " + str(len(speed_data)) + " json entries"


        if wipe and os.path.isdir(self.optflow_dir):
            print "wiping preprocessed data..."
            shutil.rmtree(self.optflow_dir)

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
        '''
        model1 = Sequential()
        model1.add(Convolution2D(3, 5,5 ,border_mode='same',input_shape=(self.DSIZE[0],self.DSIZE[1],3)))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(24, 5,5 ,border_mode='same'))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(36, 5,5 ,border_mode='same'))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(48, 3,3,border_mode='same'))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(64, 3,3,border_mode='same'))
        model1.add(Activation('relu'))
        model1.add(Flatten())
        model1.add(Dropout(0.25))
        model1.add(Dense(64))
        model1.add(Activation('relu'))
        model1.add(Dropout(0.5))

        model2 = Sequential()
        model2.add(Convolution2D(3, 5,5 ,border_mode='same',input_shape=(self.DSIZE[0],self.DSIZE[1],3)))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(24, 5,5 ,border_mode='same'))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(36, 5,5 ,border_mode='same'))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(48, 3,3,border_mode='same'))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(64, 3,3,border_mode='same'))
        model2.add(Activation('relu'))
        model2.add(Flatten())
        model2.add(Dropout(0.25))
        model2.add(Dense(64))
        model2.add(Activation('relu'))
        model2.add(Dropout(0.5))

        self.model = Sequential()
        self.model.add(Merge([model1,model2],mode='concat', concat_axis=1))
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        '''

        model1 = Sequential()
        model1.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),input_shape=(self.DSIZE[0],self.DSIZE[1],3)))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
        model1.add(Activation('relu'))
        model1.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
        model1.add(Activation('relu'))
        model1.add(Flatten())
        model1.add(Dropout(0.5))

        model2 = Sequential()
        model2.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),input_shape=(self.DSIZE[0],self.DSIZE[1],3)))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
        model2.add(Activation('relu'))
        model2.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
        model2.add(Activation('relu'))
        model2.add(Flatten())
        model2.add(Dropout(0.5))

        self.model = Sequential()
        self.model.add(Merge([model1,model2],mode='concat', concat_axis=1))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

        #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer='adam', loss='mse')
        plot(self.model, to_file='model2.png')


    def train(self,X_src,Y_src,val_split, wipe,n_epochs = 50, batch_size= 32):
        #load data
        X,Y = self.prep_data(X_src,Y_src,shuffle = False)
        Y = Y[0:-1]
        X1 = X[1:]
        X = X[0:-1]

        X = X[0:1000]
        X1 = X1[0:6000]
        Y = Y[0:6000]

        #train model
        print "Starting training"
        self.model.fit([X,X1], Y, batch_size=batch_size,
                    nb_epoch=n_epochs,validation_split=val_split)
        #save weights
        print "Done training. Saving weights"
        self.model.save_weights(self.W_FILE)

    def test(self,X_src, Y_src):
        "Starting testing"
        #load data
        X,Y = self.prep_data(X_src,Y_src,shuffle = False)
        Y = Y[0:-1]

        X1 = X[1:]
        X = X[0:-1]
        #load weights
        try:
            self.model.load_weights(self.W_FILE)
            #test the model on unseen data
            print self.model.evaluate([X,X1],Y)
            print "Done testing"
        except ValueError:
            print "Unable to load weights. Model has changed"
            print "Please retrain model"
        except IOError:
            print "Unable to load weights. No previous weights found"
            print "Please train model"

    def play(self, X_src, Y_src):
        print "loading weights"
        try:
            self.model.load_weights(self.W_FILE)
            #test the model on unseen data

        except ValueError:
            print "Unable to load weights. Model has changed"
            print "Please retrain model"
        except IOError:
            print "Unable to load weights. No previous weights found"
            print "Please train model"


        #load data
        X,Y = self.prep_data(X_src,Y_src,shuffle = False)
        X = X[np.where(Y > 0)]
        Y = Y[np.where(Y> 0)]
        Y = Y[0:-1]

        X1 = X[1:]
        X = X[0:-1]

        i = 0
        for x,x1,y in zip(X,X1,Y):
            #xin = np.array([[x],[x1]])
            #xin = np.array([[x,x1]])
            #xin = np.array([x,x1])
            pred = self.model.predict([np.array([x]) , np.array([x1])])
            print ("\rFrame: " + str(i) +
                            " Predicted: " + str(pred) +
                            " Actual: " + str(y) +
                            " Error: " + str(abs(y-pred)))
            i+=1




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file",
                        help="video file name")
    parser.add_argument("speed_file",
                        help="json speed data file name")
    parser.add_argument("--split", type=float, default=0,
                        help="percentage of train data for validation")
    parser.add_argument("--mode", choices=["train", "test", "play"], default='train',
                        help="Train or Test model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    print "Running SpeedNet by Daniel Nugent"
    net = SpeedNet()
    net.main(args)
