from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *
import glob
from torchvision import transforms
import pandas as pd
from scipy import signal
import cv2
import string
import random
import os
import pathlib



class ExtractingFrames():


    def __init__(self, datasetConfig):

        super().__init__()
        self.datasetConfig = datasetConfig

        self.sig_add = "../onedrive/General/{}Test{}/sorted/{}/{}.csv".format(self.datasetConfig["season"], self.datasetConfig["year"], self.datasetConfig["date"] + "_" + self.datasetConfig["hour"], self.datasetConfig["date"])
        self.vid_add = "../onedrive/General/{}Test{}/sorted/{}".format(self.datasetConfig["season"], self.datasetConfig["year"], self.datasetConfig["date"] + "_" + self.datasetConfig["hour"])
        self.frames_add = "../volvoData/General/{}Test{}/sorted/{}/frames".format(self.datasetConfig["season"], self.datasetConfig["year"], self.datasetConfig["date"] + "_" + self.datasetConfig["hour"])
        
        self.signals = pd.read_csv(self.sig_add)
        self.signals["date_time"] = pd.to_datetime(self.signals["date_time"])
        self.signals = self.signals.set_index('date_time')
        self.signals = self.signals.loc[~self.signals.index.duplicated(keep='first')]
        self.signals.sort_values(by='date_time', inplace=True)
        self.signals.reset_index(inplace=True)
        
        [a,b] = signal.butter(10, 0.02)
        self.signals['IsoVsLongitudinalAcceleration'] = signal.filtfilt(a, b, self.signals['IsoVsLongitudinalAcceleration'])
        self.signals["avg_mu"] = self.signals['IsoVsLongitudinalAcceleration']/9.82
        
        
        filename = glob.glob(os.path.join(self.vid_add,"*.mp4"))[0]
        self.cap = cv2.VideoCapture(filename)
        
        
        if not os.path.exists(self.frames_add):
            path = pathlib.Path(self.frames_add)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir(parents=True, exist_ok=True)
            print("Directory " , self.frames_add ,  " Created ")
        else:
            shutil.rmtree(add, ignore_errors=True)
            path = pathlib.Path(self.frames_add)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir(parents=True, exist_ok=True)
            print("Directory " , self.frames_add ,  " already exists. It is deleted and a new one created.")
        
        

    def __len__(self):
        return len(self.signals[self.signals["FrameNum"]>0])

    def __getitem__(self, idx):


        frame = self.retFrame(self.signals[self.signals["FrameNum"]>0][["FrameNum"]].iloc[idx].item())
        mu = self.signals[self.signals["FrameNum"]>0][["avg_mu"]].iloc[idx].item()

        return {'image': frame, 'label': mu}


    def retFrame(self, indexFrame):

        self.cap.set(1, indexFrame)
        ret, frame = self.cap.read()

        return frame

    
    def savingFrames(self):
        
        f = open(os.path.join(self.frames_add, "framesAdd.txt"), "w")
        f.write("fraemIdx,frameAdd,fric\n")
    
        for i in range(self.__len__()):
            frame_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".png"
            frame_add = os.path.join(self.frames_add, frame_name)
            sample = self.__getitem__(i)
            cv2.imwrite(frame_add, sample["image"])
            f.write("{},{},{}\n".format(str(i),frame_add,str(sample["label"])))
            
            
        f.close()
        
        
        