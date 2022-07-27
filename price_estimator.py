# -*- coding: utf-8 -*-
"""
Created on today

@author: ianre
"""
import os
global index
global current_file
global error

from os import listdir
from os import walk
from os.path import isfile, join

from subprocess import check_output
from subprocess import Popen, PIPE

global unique_trials 
unique_trials = {}
global total_videos
total_videos = 0

STRIDE = 15

def main():
    PE = PriceEstimator("Peg_Transfer")
    PE.countTaskFrames()


class PriceEstimator():
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"video")

    def countTaskFrames(self):   
        global unique_trials, total_videos   
        print("Analyzing:",self.imagesDir)
        total_fames = 0
        video_count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                total_videos +=1
                trial = file.replace("_right.avi","")
                trial = trial.replace("_left.avi","")
                if(trial in unique_trials.keys()):
                    continue
                else:
                    
                    with open(os.path.join(root, file), "r") as auto:                    
                        command = "C:\\FFmpeg\\bin\\ffprobe.exe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 " + auto.name
                        p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
                        rc = p.returncode

                        print(trial, video_count, " : ", int(output))
                        total_fames += int(output)
                        video_count += 1
                    unique_trials[trial] = int(output)


        print("Total Unique Trial Count: ", video_count, "out of", total_videos)
        print("Total Frame Count: ", total_fames)
        print("Labeling every ",STRIDE,"frames yields", int(total_fames/STRIDE),"labeled frames")

main()
'''
#import ffmpy

#vid_place='C:/Users/ianre/Desktop/Academica/Research/aws raw videos/JIGSAWS_needlepassing/G4_sub/*'
#mypath=r'C:/Users/ianre/Desktop/Academica/Research/aws raw videos/JIGSAWS_needlepassing/G4_sub/'
vid_place='C:/Users/ianre/Desktop/Academica/Research/aws_videos/JIGSAWS_needlepassing/*'
#mypath=r'C:/Users/ianre/Desktop/Academica/Research/aws_videos/JIGSAWS_needlepassing/'
mypath=r'C:\\Users\\ianre\\Desktop\\Academica\\Research\\aws_videos\\JIGSAWS_suturing'
#JIGSAWS_suturing

quit()

'''