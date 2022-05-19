'''
Date	MTML_position_x	MTML_position_y	MTML_position_z	MTML_orientation_x	MTML_orientation_y	MTML_orientation_z	MTML_orientation_w	MTML_gripper_angle	MTMR_position_x	MTMR_position_y	MTMR_position_z	MTMR_orientation_x	MTMR_orientation_y	MTMR_orientation_z	MTMR_orientation_w	MTMR_gripper_angle	PSML_position_x	PSML_position_y	PSML_position_z	PSML_orientation_x	PSML_orientation_y	PSML_orientation_z	PSML_orientation_w	PSML_gripper_angle	PSMR_position_x	PSMR_position_y	PSMR_position_z	PSMR_orientation_x	PSMR_orientation_y	PSMR_orientation_z	PSMR_orientation_w	PSMR_gripper_angle
'''
from asyncio import Task
from cProfile import label
from concurrent.futures import thread
import os, sys
import csv
import json
import pathlib
import math
from turtle import position
from cv2 import KeyPoint
from matplotlib.colors import cnames
import numpy as np
from PIL import Image, ImageDraw, ImageColor,ImageFont
#import krippendorff
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pkg_resources import invalid_marker
import statistics
from numpy.linalg import matrix_power
#from scipy.interpolate import interp1d
from scipy import interpolate
import shutil
import time
from collections import Counter
from scipy.interpolate import interp1d

global unique_labels
unique_labels = {}

global invalid_States
invalid_States = {}

global state_counts
state_counts = {}

global absolute_counts
absolute_counts = {}

global prev_lengths
prev_length = [0,0,0,0];
global thread_delta
thread_delta = [0,0,0,0];

global MAX_LEN
MAX_LEN = 200

'''
Thread_TL_X = []
Thread_TL_Y = []
Thread_TR_X = []
Thread_TR_Y = []
Thread_BL_X = []
Thread_BL_Y = []
Thread_BR_X = []
Thread_BR_Y = []    
'''

from itertools import accumulate

mathematicaColors = {
    "blue":"#5E81B5",
    "orange":"#E09C24",
    "red":"#EA5536",
    "purple":"#A5609D",
    "green":"#8FB131",
    "blue" :"#5e9ec8",
    "olive":"#929600",
    "terracotta":"#C56E1A",
    "yellow":"#FEC000",
}
# list of colors for the annotations
colors =["#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]
# opacity of the annotation masks. Values range from (0 to 255) Type tuple
opacity = (180,)
# radius of keypoint
radius = 3
def main():     
    dir=os.getcwd()
    '''
    try:
        task=sys.argv[1]
        #print(task)
    except:
        print("Error: no task provided \nUsage: python draw_labels.py <task>")
        available_tasks = next(os.walk(os.path.join(dir, "images")))[1]
        print("Available task images: ", available_tasks)
        available_labels = next(os.walk(os.path.join(dir, "labels")))[1]
        print("Available task labels: ", available_labels)
        sys.exit()
    '''
    global state_counts 
    global absolute_counts
    start = time.time()  

    task = "Needle_Passing"
    #I = Iterator(task) 
    absolute_counts["Labeled Frames"]  = 0
    I = Iterator(task)
    I.DrawLabelsContext()
    quit();    
   
'''
JSONInterface deals with the particular JSON format of the annotations
It's set up now to read labels as we received them from Cogito

If the JSON annotations are in a different format, you can edit the getPolygons and getKeyPoints methods
'''
class JSONInterface:    
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc    
        with open(self.json_location) as f:
            data = json.load(f)
            self.data = data
            self.meta = data['metadata']
            self.instances = data['instances']      
    '''
    Returns a list of polygons
    each polygon is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    '''
    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polygon"):                
                polygonSeries.append(instance["points"])   
                cn.append(instance["className"])
        return cn, polygonSeries
    '''
    Returns a list of PolyLines
    each polyline is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    '''
    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "point"):                
                keyPoints.append([instance['x'], instance['y']])   
                cn.append(instance["className"])
        return cn, keyPoints

    '''
    Returns a list of PolyLines
    each PolyLine is a list [x, y]
    '''
    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polyline"):                
                polylineSeries.append(instance["points"])    
                cn.append(instance["className"])
        return cn,polylineSeries

class MPInterface:
    def __init__(self,MPLoc):
        self.mp_loc = MPLoc
        self.transcript = []
        with open(self.mp_loc) as file:
            for line in file:
                #print(line.rstrip())
                self.transcript.append(line.rstrip())
    def getMP(self, index):
        #print("GetMP Matching",index)
        for i in range(1,len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if(int(l_s[1]) > index):
                return " ".join(l_s)

class ContextInterface:
    def __init__(self, ContextLoc):
        self.c_loc = ContextLoc
        self.transcript = []
        with open(self.c_loc) as file:
            for line in file:
                self.transcript.append(line.rstrip())
    def getContext(self,index):
        for i in range(1,len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if(int(l_s[0]) > index):
                #return " ".join(min(0,i-1))
                return self.transcript[max(0,i-1)]

class Iterator:
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images_pre")
        self.labelsDir = os.path.join(self.CWD, task,"annotations_pre")
        self.outputDir = os.path.join(self.CWD, task,"labeled_images")
        self.mpDir = os.path.join(self.CWD, task, "motion_primitives_combined")
        
        self.mpDir_R = os.path.join(self.CWD, task, "motion_primitives_R")
        
        self.mpDir_L = os.path.join(self.CWD, task, "motion_primitives_L")

        self.ContextDir = os.path.join(self.CWD,task,"transcriptions")

    def imageToJSON(self, file):
        fileArr = file.split(".")
        return "".join(fileArr[:-1]) + ".json"

    def getRBGA(self, hexColor):
        c = ImageColor.getcolor(hexColor, "RGB")        
        c = c + opacity
        return c

    def Centroid(self, points):
        length = len(points)
        x_arr = [];
        x_len = 0;
        y_arr = [];
        y_len = 0;
        for i in range(0,length):
            if( i %2==0):
                x_arr.append(points[i]);
                x_len=x_len+1;
            else:
                y_arr.append(points[i]);
                y_len = y_len+1;
        sum_x = np.sum(x_arr)
        sum_y = np.sum(y_arr)

        return sum_x/x_len, sum_y/y_len;
        
    def reorderPolyLines(self, polyLines):
        newPolyLines = [];
        endPoints = [];
        for p in polyLines:
            x0 = p[0]
            y0 = p[1]
            xn = p[-2]
            yn = p[-1]
            endPoints.append( [ [x0,xn],[y0,yn] ] ) # An,Bn

        
        return newPolyLines

    def DrawSingleImageKT(self, imageSource, labelSource, target, DEBUG=False):
       
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        #font = ImageFont.truetype("arial.ttf", 12)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")

        #print(polyLines)
        #return    
        img = Image.open(imageSource)
        #draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   
        polyNames_TL = []
        polyLines_TL = []
        polyNames_TR = []
        polyLines_TR = []
        polyNames_BL = []
        polyLines_BL = []
        polyNames_BR = []
        polyLines_BR = []
        for i in range(len(polyLines)):
            if("Top Left" in polyLineNames[i]):
                polyNames_TL.append(polyLineNames[i])
                polyLines_TL.append(polyLines[i])
            elif("Top Right" in polyLineNames[i]):
                polyNames_TR.append(polyLineNames[i])
                polyLines_TR.append(polyLines[i])
            elif("Bottom Left" in polyLineNames[i]):
                polyNames_BL.append(polyLineNames[i])
                polyLines_BL.append(polyLines[i])
            elif("Bottom Right" in polyLineNames[i]):
                polyNames_BR.append(polyLineNames[i])
                polyLines_BR.append(polyLines[i])
        if(len(polyLines_TL)>0):                
            kp = [polyLines_TL[0][0], polyLines_TL[0][1]]             
            self.DrawThread(polyLines_TL, polyNames_TL, kp, draw, font)
        if(len(polyLines_TR)>0):                
            kp = [polyLines_TR[0][0], polyLines_TR[0][1]]             
            self.DrawThread(polyLines_TR, polyNames_TR, kp, draw, font)
        if(len(polyLines_BL)>0):                
            kp = [polyLines_BL[0][0], polyLines_BL[0][1]]             
            self.DrawThread(polyLines_BL, polyNames_BL, kp, draw, font)
        if(len(polyLines_BR)>0):                
            kp = [polyLines_BR[0][0], polyLines_BR[0][1]]             
            self.DrawThread(polyLines_BR, polyNames_BR, kp, draw, font)

        img.save(target) # to save

    def DrawSingleImage(self, imageSource, labelSource, target, MPI, CtxI, DEBUG=False):
       
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        #font = ImageFont.truetype("arial.ttf", 12)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")
        #print(polyLines)
        #return    
        img = Image.open(imageSource)
        IDX = int(imageSource.split("_")[-1].split(".")[0])
        #draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   

        self.DrawPolygons(polygons,polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)
        #self.DrawTextTopCorner(MPI.getMP(IDX),draw,font)
        self.DrawTextTopCorner(CtxI.getContext(IDX), draw, font)

        if("Needle End" not in kpNames):
            #print("No needle for", imageSource)
            pass
        else: 
            needleEnd = KeyPoint[0]
            for i in range(len(KeyPoint)):
                if("Needle End" in kpNames[i]):
                    needleEnd = KeyPoint[i]
            if(len(polyLines) !=0):     
                self.DrawThread(polyLines, polyLineNames, needleEnd, draw, font)

        img.save(target) # to save

    def DrawSingleImageContext(self, imageSource, labelSource, target, MPI, CtxI, DEBUG=False):
       
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")
        
        img = Image.open(imageSource)
        IDX = int(imageSource.split("_")[-1].split(".")[0])
        draw = ImageDraw.Draw(img, "RGBA")   

        self.DrawPolygons(polygons,polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)
        #self.DrawTextTopCorner(MPI.getMP(IDX),draw,font) #MP
        #self.DrawTextTopCorner(CtxI.getContext(IDX), draw, font)


        if("Needle End" not in kpNames):
            #print("No needle for", imageSource)
            pass
        else: 
            needleEnd = KeyPoint[0]
            for i in range(len(KeyPoint)):
                if("Needle End" in kpNames[i]):
                    needleEnd = KeyPoint[i]
            if(len(polyLines) !=0):     
                self.DrawThread(polyLines, polyLineNames, needleEnd, draw, font)

        self.DrawTextArr([CtxI.getContext(IDX),"info about graspers", "info about objects"], draw, font)

        img.save(target) # to save

    def RenderThread_Arr(self, thread_X, thread_Y, draw, font):        
        kk=0   
        t_min = min(len(thread_X),len(thread_Y))
        thread_X = thread_X[0:t_min]
        thread_Y = thread_Y[0:t_min]       
        distance = np.cumsum(np.sqrt( np.ediff1d(thread_X, to_begin=0)**2 + np.ediff1d(thread_Y, to_begin=0)**2 ))
        distance = distance/distance[-1]        
        fx, fy = interp1d( distance, thread_X ), interp1d( distance, thread_Y )                
        alpha = np.linspace(0, 1, 10)
        x_regular, y_regular = fx(alpha), fy(alpha)
        for jj in range(0,len(thread_X)):           
            draw.line((thread_X[kk],
                    thread_Y[kk],
                    thread_X[kk+1],
                    thread_Y[kk+1]), fill=(255, 0, 0, 127), width=2) 
            kk+=1
            if(kk>=len(thread_X)-1): break
        for i in range(len(x_regular)): # draws each KeyPoint
            x = x_regular[i]
            y = y_regular[i]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            #c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList, fill=(0,0,0))
    
    def dist(self, A, B):
        return math.sqrt( (A[0] - B[0] )**2+( A[1]-B[1])**2 )

    def distNeedle(self, ax, ay,needleEnd ):
        return math.sqrt( (ax - needleEnd[0] )**2+( ay - needleEnd[1] )**2 )

    def PutTheadInOrder(self,polyLines_M,kp):
        thread_X = []
        thread_Y = []  
        polyLines = polyLines_M.copy()
        while(len(polyLines) > 0):    
            index_of_closest_thread_arr = 0; 
            position_of_closest = 0; # 0 for cannonical 0, 1 for end
            thread_idx_0_distances = []
            thread_idx_1_distances = [] 

            for i in range(len(polyLines)):
                l = len(polyLines[i])
                fx = polyLines[i][0]
                fy = polyLines[i][1]
                lx = polyLines[i][l-2]
                ly = polyLines[i][l-1]
                thread_idx_0_distances.append(self.distNeedle(fx,fy,kp))
                thread_idx_1_distances.append(self.distNeedle(lx,ly,kp))       

            min_0 = min(thread_idx_0_distances) # min of all distances between last end
            min_1 = min(thread_idx_1_distances)

            idx_0 = thread_idx_0_distances.index(min_0)
            idx_1 = thread_idx_1_distances.index(min_1)
            if(min_0 < min_1):
                position_of_closest = 0
                index_of_closest_thread_arr = idx_0
            else:
                position_of_closest = 1
                index_of_closest_thread_arr = idx_1
            
            shortest = polyLines[index_of_closest_thread_arr]
            if(position_of_closest ==0):
                for j in range(0,len(shortest),2):
                    thread_X.append(shortest[j])
                    thread_Y.append(shortest[j+1])
                    kp = [shortest[j],shortest[j+1]]
            else:
                ll = len(shortest)
                for j in range(0,len(shortest),2):
                    thread_X.append(shortest[ll-j-2])
                    thread_Y.append(shortest[ll-j-1])
                    kp = [shortest[ll-j-2],shortest[ll-j-1]]


            
            del polyLines[index_of_closest_thread_arr]



        return thread_X, thread_Y

    def DrawThread(self,polyLines, polyLineNames, needleEnd, draw, font):  
        thread_X = []
        thread_Y = []
        if(len(polyLines)<2):
            for i in range(len(polyLines)):
                l = len(polyLines)
                for j in range(0,len(polyLines[i]),2):
                    thread_X.append(polyLines[i][j])
                    thread_Y.append(polyLines[i][j+1])
        else:
            thread_X, thread_Y = self.PutTheadInOrder(polyLines,needleEnd)
      

        '''
        for i in range(len(polyLines)):
            l = len(polyLines)
            c = self.getRBGA(colors[-i])
            k=0
            for j in range(0,len(polyLines[l-i-1]),2):
                thread_X.append(polyLines[l-i-1][j])
                thread_Y.append(polyLines[l-i-1][j+1])
        '''  
        for i in range(len(polyLines)):
            c = self.getRBGA(colors[-i])
            k=0          

            for j in range(0,len(polyLines[i])):
                #if(k+3>=len(polyLines[i])):
                #    
                draw.line(( polyLines[i][k],
                            polyLines[i][k+1],
                            polyLines[i][k+2],
                            polyLines[i][k+3]), fill=c, width=9) 
                x = polyLines[i][k]
                y = polyLines[i][k+1]          
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                x = polyLines[i][k+2]
                y = polyLines[i][k+3]     
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                k+=2
                if(k>=len(polyLines[i])-2): break
            draw.text((polyLines[i][0],polyLines[i][1]),polyLineNames[i]+str(i),(255,255,255),font=font)   

        self.RenderThread_Arr(thread_X, thread_Y, draw, font)
            
    def DrawKeyPoints(self,KeyPoint, kpNames, polygons,draw,font):
        for i in range(len(KeyPoint)): # draws each KeyPoint
            x = KeyPoint[i][0]
            y = KeyPoint[i][1]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList, fill=c)
            draw.text((x-radius*2, y-radius),kpNames[i]+str(i),(255,255,255),font=font)

    def DrawTextTopCorner(self,MPI_str,draw,font):
        if(MPI_str is None): 
            return           
        x = 50
        y = 50
        leftUpPoint = (x-radius, y-radius)
        rightDownPoint = (x+radius, y+radius)
        twoPointList = [leftUpPoint, rightDownPoint]
        #c = self.getRBGA(colors[(len(polygons))])
        #draw.ellipse(twoPointList, fill=c)
        draw.text((x-radius*2, y-radius),MPI_str,(255,255,255),font=font)

    def DrawTextArr(self,strArr,draw,font):
        if(len(strArr) == 0 or strArr[0] is None): 
            return
        offset = 1
        for s in strArr:
            x = 10
            y = 15 * offset; 
            draw.text( (x,y),s,(255,255,255),font=font);
            offset+=1

    def DrawPolygons(self, polygons,polyNames,draw,font):
        for i in range(len(polygons)):
            #if("Ring" in polyNames[i]):
            c = self.getRBGA(colors[i])
                #print("Poly1:",polygons[i])
            draw.polygon(polygons[i], fill=c) #,outline='#EA5536')     
                ########## CENTER POINT
            x_c, y_c = self.Centroid(polygons[i])          
            leftUpPoint = (x_c-radius, y_c-radius)
            rightDownPoint = (x_c+radius, y_c+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList,outline=1, fill=c)            
                # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)

    def DrawLabelsContext(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                #if "Suturing_S02_T05" not in os.path.basename(root):
                #    continue
                print("Proc:", os.path.basename(root),file+".txt" )
                Bname = os.path.basename(root)

                MP_comb = os.path.join(self.mpDir,Bname+".txt")
                #print(MP_comb)
                MPI = MPInterface(MP_comb)

                Context_comb = os.path.join(self.ContextDir,Bname+".txt")
                #print(MP_comb)
                CtxI = ContextInterface(Context_comb)
                '''
                If we replace "images" by "labels" then the image source should be the same as the label source, 
                which is the same as the output destination
                '''
                imageRoot = root
                #labelRoot = self.getDirectory(root,"labels")
                labelRoot = root.replace("images_pre","annotations_pre")
                #outputRoot =  self.getDirectory(root,"output")
                outputRoot = root.replace("images_pre","labeled_images")

                imageSource = os.path.join(imageRoot, file)
                labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                outputDest = os.path.join(outputRoot, file)

                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)

                #if os.path.exists(outputDest):
                #    os.remove(outputDest)

                if not os.path.exists(labelSource):
                    #print("label not found for ",imageSource)
                    continue
                else:
                    #self.DrawLabel(imageSource,labelSource,outputDest)
                    if("Knot" in self.task):
                        self.DrawSingleImageKT(imageSource,labelSource,outputDest)
                    else:
                        self.DrawSingleImageContext(imageSource,labelSource,outputDest, MPI, CtxI)

                count += 1
                
        print(count,"images processed!")

    def DrawLabels(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                #if "Suturing_S02_T05" not in os.path.basename(root):
                #    continue
                print("Proc:", os.path.basename(root),file+".txt" )
                Bname = os.path.basename(root)

                MP_comb = os.path.join(self.mpDir,Bname+".txt")
                print(MP_comb)
                MPI = MPInterface(MP_comb)


                '''
                If we replace "images" by "labels" then the image source should be the same as the label source, 
                which is the same as the output destination
                '''
                imageRoot = root
                #labelRoot = self.getDirectory(root,"labels")
                labelRoot = root.replace("images_pre","annotations_pre")
                #outputRoot =  self.getDirectory(root,"output")
                outputRoot = root.replace("images_pre","labeled_images")

                imageSource = os.path.join(imageRoot, file)
                labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                outputDest = os.path.join(outputRoot, file)

                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)

                #if os.path.exists(outputDest):
                #    os.remove(outputDest)

                if not os.path.exists(labelSource):
                    #print("label not found for ",imageSource)
                    continue
                else:
                    #self.DrawLabel(imageSource,labelSource,outputDest)
                    if("Knot" in self.task):
                        self.DrawSingleImageKT(imageSource,labelSource,outputDest)
                    else:
                        self.DrawSingleImage(imageSource,labelSource,outputDest, MPI)

                count += 1
                
        print(count,"images processed!")
       

main();

