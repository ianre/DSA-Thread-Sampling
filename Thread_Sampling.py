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
    

    task = "Knot_Tying"
    #I = Iterator(task) 
    absolute_counts["Labeled Frames"]  = 0
    I = Iterator(task)
    I.DrawLabels()

    P = ImageProc(task);
    #P.addAllVisData()
    P.addAllVisDataKT()
    
    #P.prepareImages()
    #P.addVisData();
    #P.testInterpol();
    '''
    P.prepareAnnotations();
    P2 = ImageProc("Suturing");
    P2.prepareAnnotations();
    P3 = ImageProc("Needle_Passing");
    P3.prepareAnnotations();
    '''

    '''
    print("\tOccurrences per frame:")
    state_counts = dict(sorted(state_counts.items()))
    for k, v in state_counts.items(): print(k, v)
    print("\tTotal occurrences:")
    #print(absolute_counts)
    absolute_counts = dict(sorted(absolute_counts.items()))
    for k, v in absolute_counts.items(): print(k, v)
    '''

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


class ImageProc:
    def __init__(self, task):

        self.CWD = os.path.dirname(os.path.realpath(__file__))
        self.task = task;
        self.labelDir = os.path.join(self.CWD, "cogito_labels")
        self.imagesDir = os.path.join(self.CWD, "cogito_images")
        self.taskDir = os.path.join(self.labelDir,task);
        self.taskDirButImages = os.path.join(self.imagesDir,task)
        self.kinDir = os.path.join(self.CWD, task)
        self.kinSource = os.path.join(self.kinDir, "kinematics")
        self.kinLabelDir = os.path.join(self.kinDir, "annotations_pre")
        self.imgLabelDir = os.path.join(self.kinDir, "images_pre")
        self.kinOutput = os.path.join(self.kinDir, "output")
        self.kinOutput_Ext = os.path.join(self.kinDir, "output_ext")
        self.recordsDir = os.path.join(self.CWD,"slicing_records")
        self.recordsLoc = os.path.join(self.recordsDir,task+"_Record.json")
        self.labelDict = {}
        with open(self.recordsLoc) as f:
            data = json.load(f)
        self.RecordData = data;

    def save(self, x_file, x_lines):
        with open(x_file, 'w+') as f:
            for item in x_lines:
                f.write("%s\n" % item)
    
    def interpolate(self, l_s):
        n = [4, 0, 0, 6, 0, 8, 0, 0, 0, 3]
        starts = accumulate(range(len(l_s)),lambda a,b: b if l_s[b] else a)
        ends   = [*accumulate(reversed(range(len(l_s))),lambda a,b: b if l_s[b] else a)][::-1]
        inter  = [ l_s[i] or l_s[s]+(l_s[e]-l_s[s])*(i-s)/(e-s) for i,(s,e) in enumerate(zip(starts,ends)) ]
        return inter;

    def extendFirstLast(self,l_s):
        for i in range(len(l_s)):
            if(l_s[i] >0):
                l_s[0]=l_s[i];
                break;
        for i in range(len(l_s)-1,-1,-1):
            if(l_s[i] >0):
                l_s[len(l_s)-1]=l_s[i];
                break;

        return l_s;

    def RenameDirs(self):
        print(self.task)
        count= 0
        for key in self.labelDict.keys():
            print(key,"=",self.getSubject(key)) #self.labelDict[key]
            for kkey in self.labelDict[key].keys():
                print("\t",int(kkey),len(self.labelDict[key][kkey].keys()))
                count=count+1;
        print("Total:",count)

    def getFormattedName(self, subject, trial ):
        newName = self.task+"_S0"+str(self.getSubject(subject))+"_T"+str(trial)  #+"_"+vid
        return newName

    # FormatAnnotations looks at the cogito labels under a particular task
    def FormatAnnotations(self):
        count = 0;
        for root, dirs, files in os.walk(self.taskDir):# for cogito annotations
            for file in files:
                if "frame" not in file:
                    continue
                imageRoot = root
                lastRoot = os.path.basename(root)
                subj, trial, trialStr, StartFrame, EndFrame = self.decode(str(lastRoot))

                frameFilename = os.path.join(root,file)
                newFileDir = os.path.join(self.kinLabelDir,self.getFormattedName(subj, trialStr))
                newFilename = self.getFormattedName(subj, trialStr)+"_"+StartFrame+"_"+EndFrame+".json"
                #filename = os.path.basename(item[0])                
                #copyfile(item[0], os.path.join("/Users/username/Desktop/testPhotos", filename))
                #print("copy:",frameFilename,newFileDir,newFilename)
                if not os.path.exists(newFileDir):
                    os.makedirs(newFileDir)
                fullName = os.path.join(newFileDir, newFilename)
                shutil.copyfile(frameFilename, fullName)  
                count += 1
        print(count,"files processed!")

    def fromOurFormatGetS00T(self, trial):
        # trial is Knot_Tying_S02_T01.csv
        pass;

    def getFrameNumber(self,file):
        return file.replace("frame_","").replace(".png___objects.json","")
   
    def getFrameNumberIMG(self,file):
        return file.replace("frame_","").replace(".png","")
        #print("file is:"file)
    def pad4zeroes(self, f_n):
        f_s = str(f_n)
        
        return f_s.zfill(4) 

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

    #return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)
    def MidPoint(self, Needle_Tip, Needle_End):
        #print("Needle_Tip",Needle_Tip)
        #print("Needle End", Needle_End)
        #print( str((Needle_Tip[0]+Needle_End[0])/2), str((Needle_Tip[1]+Needle_End[1])/2) )

        return (Needle_Tip[0]+Needle_End[0])/2, (Needle_Tip[1]+Needle_End[1])/2

    def addThisAFileToKin(self,labelSource, shortName,index,pre_inter_labels):
        #L_Gripper_X
        #L_Gripper_Y
        #R_Gripper_X
        #L_Gripper_Y

        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons();
        kpNames, KeyPoint = J.getKeyPoints();
        polyLineNames, polyLines = J.getPolyLines();
        #print("\t\tFor",shortName,"polyg:",polyNames,"Key:",kpNames,"lines",polyLineNames)
        if("Left Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Left Grasper")])  
            pre_inter_labels[index] +=","+str(x_c)
            pre_inter_labels[index] +=","+str(y_c)            
        else:
            pre_inter_labels[index] +=","+str(0)
            pre_inter_labels[index] +=","+str(0) 

        if("Right Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Right Grasper")])  
            pre_inter_labels[index] +=","+str(x_c)
            pre_inter_labels[index] +=","+str(y_c)
        else:
            pre_inter_labels[index] +=","+str(0)
            pre_inter_labels[index] +=","+str(0)   
        return pre_inter_labels

    def count_and_add(self, className, names):
        global absolute_counts
        if(className in names): #'Needle Mask', 'Needle Mask', 'Needle Mask', 'Needle Mask', 'Left Grasper', 'Right Grasper']
            if(className in absolute_counts):
                absolute_counts[className] +=1
            else:
                absolute_counts[className] = 1

    def just_add(self, className, names):
        global state_counts    
        if(className in state_counts):
            state_counts[className] +=1
        else:
            state_counts[className] = 1

    def generateAnnotationSequencesNP(self,labelSource, shortName,index,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y): 
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        global state_counts
        global absolute_counts
        if(index >= len(L_G_X) or index >= len(L_G_Y) ):
            return L_G_X, L_G_Y, R_G_X, R_G_Y, MID_N_X, MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y
        
        absolute_counts["Labeled Frames"] +=1;
        for name in polyNames:
            self.count_and_add(name,polyNames)

        for name in kpNames:
            self.count_and_add(name, kpNames)

        for name in polyLineNames:
            self.count_and_add(name, polyLineNames)

        polyNames_s = Counter(polyNames)
        for name in polyNames_s:
            self.just_add(name+"_"+str(polyNames_s[name]), polyNames)
            #print(name,polyNames_s[name])

        kpNames_s = Counter(kpNames)
        for name in kpNames_s:
            self.just_add(name+"_"+str(kpNames_s[name]), kpNames)
            #print(name,kpNames_s[name])

        polyLineNames_s = Counter(polyLineNames)
        for name in polyLineNames_s:
            self.just_add(name+"_"+str(polyLineNames_s[name]), polyLineNames)
            #print(name,polyLineNames_s[name])
       
        #print("\t\t",shortName,",",polyNames,",",kpNames,",",polyLineNames)
        if("Left Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Left Grasper")])  
            L_G_X[index] = x_c
            L_G_Y[index] = y_c       
        else:
            L_G_X[index] = 0
            L_G_Y[index] = 0  
        if("Right Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Right Grasper")])  
            R_G_X[index] = x_c
            R_G_Y[index] = y_c 
        else:
            R_G_X[index] = 0
            R_G_Y[index] = 0   
        if("Needle Tip" in kpNames and "Needle End" in kpNames):
            x_c, y_c = self.MidPoint(KeyPoint[kpNames.index("Needle Tip")],KeyPoint[kpNames.index("Needle End")])  
            MID_N_X[index] = x_c
            MID_N_Y[index] = y_c 
        else:
            MID_N_X[index] = 0
            MID_N_Y[index] = 0  
        #R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y): 

        if("Ring_4" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Ring_4")])  
            R_4_X[index] = x_c
            R_4_Y[index] = y_c       
        else:
            R_4_X[index] = 0
            R_4_Y[index] = 0  
        if("Ring_5" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Ring_5")])  
            R_5_X[index] = x_c
            R_5_Y[index] = y_c       
        else:
            R_5_X[index] = 0
            R_5_Y[index] = 0  
        if("Ring_6" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Ring_6")])  
            R_6_X[index] = x_c
            R_6_Y[index] = y_c       
        else:
            R_6_X[index] = 0
            R_6_Y[index] = 0  
        if("Ring_7" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Ring_7")])  
            R_7_X[index] = x_c
            R_7_Y[index] = y_c       
        else:
            R_7_X[index] = 0
            R_7_Y[index] = 0  


        return L_G_X, L_G_Y, R_G_X, R_G_Y, MID_N_X, MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y

    def generateAnnotationSeqThread(self,labelSource, shortName,index,TRD_X, TRD_Y):
        J = JSONInterface(labelSource)
        polyLineNames, polyLines = J.getPolyLines();

        global state_counts
        global absolute_counts
        # need to store 5 points in each one of the thread classes. 
        T_TL_X = [] # array of X points of all points in class Thread Top Left for this annotation
        T_TL_Y = []
        T_TR_X = []
        T_TR_Y = []
        T_BL_X = []
        T_BL_Y = []
        T_BR_X = []
        T_BR_Y = []  

        if(len(polyLines) >= 4 and True == False):
            print("labelSource:",labelSource)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            polyIndex = 0
            for poly in polyLines:
                currPoly_X = []
                currPoly_Y = []
                for i in range(0,len(poly),2):
                    currPoly_X.append(poly[i])
                    currPoly_Y.append(poly[i+1])
                #colors = np.random.rand()*10
                ax1.scatter(currPoly_X, currPoly_Y, s=10, c=np.random.rand(3,), marker="s", label="index:"+str(polyIndex))
                #ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
                polyIndex+=1
            plt.legend(loc='upper left');
            plt.show()

        for i in range(len(polyLines)):           
            if("Top Left" in polyLineNames[i]):
                for j in range(0,len(polyLines[i]),2):
                    T_TL_X.append(polyLines[i][j])
                    T_TL_Y.append(polyLines[i][j+1])
            elif("Top Right" in polyLineNames[i]):
                for j in range(0,len(polyLines[i]),2):
                    T_TR_X.append(polyLines[i][j])
                    T_TR_Y.append(polyLines[i][j+1])
            elif("Bottom Left" in polyLineNames[i]):
                for j in range(0,len(polyLines[i]),2):
                    T_BL_X.append(polyLines[i][j])
                    T_BL_Y.append(polyLines[i][j+1])
            elif("Bottom Right" in polyLineNames[i]):
                for j in range(0,len(polyLines[i]),2):
                    T_BR_X.append(polyLines[i][j])
                    T_BR_Y.append(polyLines[i][j+1])

        T_X = [T_TL_X,T_TR_X,T_BL_X,T_BR_X]
        T_Y = [T_TL_Y,T_TR_Y,T_BL_Y,T_BR_Y]

        for i in range(len(T_Y)): # for each class in T_X, T_Y
            kk=0   
            t_min = min(len(T_X[i]),len(T_Y[i]))
            #print("lens:",len(thread_X),len(thread_Y),"min:",t_min)
            T_X[i] = T_X[i][0:t_min]
            T_Y[i]  = T_Y[i] [0:t_min]
            if t_min!=0:
                distance = np.cumsum(np.sqrt( np.ediff1d(T_X[i], to_begin=0)**2 + np.ediff1d(T_Y[i], to_begin=0)**2 ))

                global prev_length, thread_delta
                old_prev_length = prev_length[i];
                prev_length[i] = distance[-1];   
                #old_prev_length = prev_length[i];   
                thread_delta[i] = abs(prev_length[i] - old_prev_length)
                if(thread_delta[i] >150 and old_prev_length !=0):
                    for sample5 in range(0,5):
                        TRD_X[i][sample5][index] = 0;
                        TRD_Y[i][sample5][index] = 0;
                    continue;
                
                distance = distance/distance[-1]
                fx, fy = interp1d( distance, T_X[i] ), interp1d( distance, T_Y[i] )                
                alpha = np.linspace(0, 1, 5) # sample 5 points
                x_regular, y_regular = fx(alpha), fy(alpha)
                if(index>=len(TRD_X[0][0]) or index>=len(TRD_Y[0][0])): # i is [0,3], ii should be [0,4] 
                    print("danger")
                for ii in range(len(x_regular)): # draws each KeyPoint
                    x = x_regular[ii]
                    y = y_regular[ii]  
                    try:
                        TRD_X[i][ii][index] = x
                        TRD_Y[i][ii][index] = y   
                    except Exception as e:
                        print(e)
                        continue
                    
                '''
                for i in range(len(x_regular)): # draws each KeyPoint
                    x = x_regular[i]
                    y = y_regular[i]            
                    leftUpPoint = (x-radius, y-radius)
                    rightDownPoint = (x+radius, y+radius)
                    twoPointList = [leftUpPoint, rightDownPoint]
                    #c = self.getRBGA(colors[i+(len(polygons))])
                    draw.ellipse(twoPointList, fill=(0,0,0))
                '''
            else:
                #print("EVER HERE?")
                for sample5 in range(0,5):
                    TRD_X[i][sample5][index] = 0;
                    TRD_Y[i][sample5][index] = 0;


        return TRD_X, TRD_Y

    def generateAnnotationSequences(self,labelSource, shortName,index,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y): 
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        global state_counts
        global absolute_counts
        if(index >= len(L_G_X) or index >= len(L_G_Y) ):
            return L_G_X, L_G_Y, R_G_X, R_G_Y, MID_N_X, MID_N_Y
        
        #print("polyNames",polyNames)
        #print("kpNames",kpNames)
        #print("polyLineNames",polyLineNames) #['Bottom Right Thread', 'Bottom Left Thread', 'Top Left Thread', 'Top Right Thread']
        
        #if("Needle Mask" in polyNames): #'Needle Mask', 'Needle Mask', 'Needle Mask', 'Needle Mask', 'Left Grasper', 'Right Grasper']
        #    if("Needle Mask" in state_counts):
        #        state_counts["Needle Mask"] +=1
        #    else:
        #        state_counts["Needle Mask"] = 1
        absolute_counts["Labeled Frames"] +=1;
        for name in polyNames:
            self.count_and_add(name,polyNames)

        for name in kpNames:
            self.count_and_add(name, kpNames)

        for name in polyLineNames:
            self.count_and_add(name, polyLineNames)

        polyNames_s = Counter(polyNames)
        for name in polyNames_s:
            self.just_add(name+"_"+str(polyNames_s[name]), polyNames)
            #print(name,polyNames_s[name])

        kpNames_s = Counter(kpNames)
        for name in kpNames_s:
            self.just_add(name+"_"+str(kpNames_s[name]), kpNames)
            #print(name,kpNames_s[name])

        polyLineNames_s = Counter(polyLineNames)
        for name in polyLineNames_s:
            self.just_add(name+"_"+str(polyLineNames_s[name]), polyLineNames)
            #print(name,polyLineNames_s[name])
       
        #print("\t\t",shortName,",",polyNames,",",kpNames,",",polyLineNames)
        if("Left Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Left Grasper")])  
            L_G_X[index] = x_c
            L_G_Y[index] = y_c       
        else:
            L_G_X[index] = 0
            L_G_Y[index] = 0  
        if("Right Grasper" in polyNames):
            x_c, y_c = self.Centroid(polygons[polyNames.index("Right Grasper")])  
            R_G_X[index] = x_c
            R_G_Y[index] = y_c 
        else:
            R_G_X[index] = 0
            R_G_Y[index] = 0   
        if("Needle Tip" in kpNames and "Needle End" in kpNames):
            x_c, y_c = self.MidPoint(KeyPoint[kpNames.index("Needle Tip")],KeyPoint[kpNames.index("Needle End")])  
            MID_N_X[index] = x_c
            MID_N_Y[index] = y_c 
        else:
            MID_N_X[index] = 0
            MID_N_Y[index] = 0  


        return L_G_X, L_G_Y, R_G_X, R_G_Y, MID_N_X, MID_N_Y

    def addAllVisDataKT(self):
        all_kin_data = self.getKinLines();
        countTotal = 0;
        countMissing = 0;
        for trial in all_kin_data:
            trial_s = trial.replace(".csv","")
            thisTrialDir = os.path.join(self.kinLabelDir,trial_s)
            trialKin = all_kin_data[trial]
            #newFileName = os.path.join(self.kinOutput,trial_s)
            newFileName = os.path.join(self.kinOutput_Ext,trial_s)

            if not os.path.exists(thisTrialDir):
                #os.makedirs(newFileDir)
                print("No Annotations for ",thisTrialDir)
                countMissing = countMissing+1;
                continue;
            else: 
                # make a list with 0 for no annotation
                # write self.interpolate(list) to the file
                numAnnotations = 0;
                trialLength = len(trialKin)
                pre_inter_labels = [];

                for i in range(trialLength+1):
                    #pre_inter_labels.append(trialKin[i]);
                    pre_inter_labels.append(0);
                #for i in range(trialLength):
                #"Thread_TL_1_X"
                trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y"
                for g in range(1,6):
                    trialKin[0]+=",Thread_TL_"+str(g)+"_X,Thread_TL_"+str(g)+"_Y"
                for g in range(1,6):
                    trialKin[0]+=",Thread_TR_"+str(g)+"_X,Thread_TR_"+str(g)+"_Y"
                for g in range(1,6):
                    trialKin[0]+=",Thread_BL_"+str(g)+"_X,Thread_BL_"+str(g)+"_Y"
                for g in range(1,6):
                    trialKin[0]+=",Thread_BR_"+str(g)+"_X,Thread_BR_"+str(g)+"_Y"

               
                T_TL_X = []
                T_TL_Y = []
                T_TR_X = []
                T_TR_Y = []
                T_BL_X = []
                T_BL_Y = []
                T_BR_X = []
                T_BR_Y = []                
                for i in range(0,5):
                    T_TL_X.append(pre_inter_labels.copy())
                    T_TL_Y.append(pre_inter_labels.copy())
                    T_TR_X.append(pre_inter_labels.copy())
                    T_TR_Y.append(pre_inter_labels.copy())
                    T_BL_X.append(pre_inter_labels.copy())
                    T_BL_Y.append(pre_inter_labels.copy())
                    T_BR_X.append(pre_inter_labels.copy())
                    T_BR_Y.append(pre_inter_labels.copy())
                    
                T_X = [T_TL_X,T_TR_X,T_BL_X,T_BR_X] # T_X[thread class][point number][kin index]
                T_Y = [T_TL_Y,T_TR_Y,T_BL_Y,T_BR_Y]
            
                MID_N_X = pre_inter_labels.copy()
                MID_N_Y = pre_inter_labels.copy()
                L_G_X = pre_inter_labels.copy()
                L_G_Y = pre_inter_labels.copy()
                R_G_X = pre_inter_labels.copy()
                R_G_Y = pre_inter_labels.copy()               

                for root, dirs, files in os.walk(thisTrialDir):
                    for file in files:          
                        if "frame" not in file:
                            continue
                        f_n = file.replace(".json","").replace("frame_","")
                        f_n = int(f_n)
                        #trialKin[f_n] = trialKin[f_n]+",Value"

                        numAnnotations= numAnnotations+1
                        thisKinFileName = os.path.join(thisTrialDir,file)
                        #pre_inter_labels = self.addThisAFileToKin(thisKinFileName,trial,f_n,pre_inter_labels)                       
                        #if(self.task == "Needle_Passing"):
                        #    L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y = self.generateAnnotationSequencesNP(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y)
                        
                        # generateAnnotationSequences(self,labelSource, shortName,index,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y)
                        L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y = self.generateAnnotationSequences(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y)
                        # generateAnnotationSeqThread(self,labelSource, shortName,index,TRD_X, TRD_Y):
                        T_X,T_Y = self.generateAnnotationSeqThread(thisKinFileName,trial,f_n,T_X,T_Y )
                
                L_G_X = self.extendFirstLast(L_G_X)
                L_G_Y = self.extendFirstLast(L_G_Y)
                R_G_X = self.extendFirstLast(R_G_X)
                R_G_Y = self.extendFirstLast(R_G_Y)
                

                L_G_X = self.interpolate(L_G_X)
                L_G_Y = self.interpolate(L_G_Y)
                R_G_X = self.interpolate(R_G_X)
                R_G_Y = self.interpolate(R_G_Y)
                for classID in range(0,4): # for each of the 5 classes
                    #T_X[i] = self.extendFirstLast(T_X[i])
                    #T_Y[i] = self.extendFirstLast(T_Y[i]) 
                    for pointID in range(0,5): # for 5 points
                        T_X[classID][pointID] = self.extendFirstLast(T_X[classID][pointID])
                        T_Y[classID][pointID] = self.extendFirstLast(T_Y[classID][pointID])
                for classID in range(0,4): # for each of the 5 classes
                    #T_X[i] = self.extendFirstLast(T_X[i])
                    #T_Y[i] = self.extendFirstLast(T_Y[i]) 
                    for pointID in range(0,5): # for 5 points
                        T_X[classID][pointID] = self.interpolate(T_X[classID][pointID])
                        T_Y[classID][pointID] = self.interpolate(T_Y[classID][pointID])
                '''
                if(self.task == "Needle_Passing"):
                    R_4_X = self.extendFirstLast(R_4_X)
                    R_4_Y = self.extendFirstLast(R_4_Y)
                    R_5_X = self.extendFirstLast(R_5_X)
                    R_5_Y = self.extendFirstLast(R_5_Y)
                    R_6_X = self.extendFirstLast(R_6_X)
                    R_6_Y = self.extendFirstLast(R_6_Y)
                    R_7_X = self.extendFirstLast(R_7_X)
                    R_7_Y = self.extendFirstLast(R_7_Y)

                    R_4_X = self.interpolate(R_4_X)
                    R_4_Y = self.interpolate(R_4_Y)
                    R_5_X = self.interpolate(R_5_X)
                    R_5_Y = self.interpolate(R_5_Y)
                    R_6_X = self.interpolate(R_6_X)
                    R_6_Y = self.interpolate(R_6_Y)
                    R_7_X = self.interpolate(R_7_X)
                    R_7_Y = self.interpolate(R_7_Y)
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i])+","+str(R_4_X[i])+","+str(R_4_Y[i])+","+str(R_5_X[i])+","+str(R_5_Y[i])+","+str(R_6_X[i])+","+str(R_6_Y[i])+","+str(R_7_X[i])+","+str(R_7_Y[i]);
                '''

                for i in range(1,trialLength):
                    trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i]);
                    #trialKin[i] = ","+str(T_X[0][i])+","+str(T_Y[0][i]);
                    for classID in range(0,4): # for each of the 5 classes
                        for pointID in range(0,5): # for 5 points
                            trialKin[i] += ","+str(T_X[classID][pointID][i])+","+str(T_Y[classID][pointID][i])
                

            self.save(newFileName+".csv",trialKin)
            print("Saving",newFileName, "with", numAnnotations,"/",len(trialKin)," # ",str(numAnnotations/len(trialKin)),"labeled")


    def addAllVisData(self):
        all_kin_data = self.getKinLines();
        countTotal = 0;
        countMissing = 0;
        for trial in all_kin_data:
            trial_s = trial.replace(".csv","")
            thisTrialDir = os.path.join(self.kinLabelDir,trial_s)
            trialKin = all_kin_data[trial]
            #newFileName = os.path.join(self.kinOutput,trial_s)
            newFileName = os.path.join(self.kinOutput_Ext,trial_s)

            if not os.path.exists(thisTrialDir):
                #os.makedirs(newFileDir)
                print("No Annotations for ",thisTrialDir)
                countMissing = countMissing+1;
                continue;
            else: 
                # make a list with 0 for no annotation
                # write self.interpolate(list) to the file
                numAnnotations = 0;
                trialLength = len(trialKin)
                pre_inter_labels = [];

                for i in range(trialLength):
                    #pre_inter_labels.append(trialKin[i]);
                    pre_inter_labels.append(0);
                #for i in range(trialLength):
                if(self.task=="Needle_Passing"):
                    trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y,Needle_X,Needle_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y"
                elif (self.task=="Knot_Tying"): 
                    trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y"
                else:
                    trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y,Needle_X,Needle_Y"
                
                # rings:
                '''
                R_arr_X = []
                R_arr_Y = []
                for i in range(0,4):
                    R_arr_X.append(pre_inter_labels.copy())
                    R_arr_Y.append(pre_inter_labels.copy())
                '''

                # Thread
                # 0 for top left, 1 for
                '''
                TL_arr_X = []
                TL_arr_Y = []
                TR_arr_X = []
                TR_arr_Y = []
                BL_arr_X = []
                BL_arr_Y = []
                BR_arr_X = []
                BR_arr_Y = []
                TRD_X = []
                TRD_Y = []
                for i in range(0,5):
                    TL_arr_X.append(pre_inter_labels.copy())
                    TL_arr_Y.append(pre_inter_labels.copy())
                    TR_arr_X.append(pre_inter_labels.copy())
                    TR_arr_Y.append(pre_inter_labels.copy())
                    BL_arr_X.append(pre_inter_labels.copy())
                    BL_arr_Y.append(pre_inter_labels.copy())
                    BR_arr_X.append(pre_inter_labels.copy())
                    BR_arr_Y.append(pre_inter_labels.copy())
                    TRD_X.append(pre_inter_labels.copy())
                    TRD_Y.append(pre_inter_labels.copy())
                '''


                L_G_X = pre_inter_labels.copy()
                L_G_Y = pre_inter_labels.copy()
                R_G_X = pre_inter_labels.copy()
                R_G_Y = pre_inter_labels.copy()

                MID_N_X = pre_inter_labels.copy()
                MID_N_Y = pre_inter_labels.copy()

                R_4_X = pre_inter_labels.copy()
                R_4_Y = pre_inter_labels.copy()
                R_5_X = pre_inter_labels.copy()
                R_5_Y = pre_inter_labels.copy()
                R_6_X = pre_inter_labels.copy()
                R_6_Y = pre_inter_labels.copy()
                R_7_X = pre_inter_labels.copy()
                R_7_Y = pre_inter_labels.copy()
               

                for root, dirs, files in os.walk(thisTrialDir):
                    for file in files:          
                        if "frame" not in file:
                            continue
                        f_n = file.replace(".json","").replace("frame_","")
                        f_n = int(f_n)
                        #trialKin[f_n] = trialKin[f_n]+",Value"

                        numAnnotations= numAnnotations+1
                        thisKinFileName = os.path.join(thisTrialDir,file)
                        #pre_inter_labels = self.addThisAFileToKin(thisKinFileName,trial,f_n,pre_inter_labels)
                        '''
                        if(self.task == "Needle_Passing"):
                            L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y = self.generateAnnotationSequences(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y)
                            TRD_X, TRD_Y = self.generateAnnotationSeqThread(thisKinFileName,trial,f_n,TRD_X, TRD_Y)
                        else:
                            continue
                        '''
                        if(self.task == "Needle_Passing"):
                            L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y = self.generateAnnotationSequencesNP(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y)
                        else:
                            L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y = self.generateAnnotationSequences(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y)
                
                L_G_X = self.extendFirstLast(L_G_X)
                L_G_Y = self.extendFirstLast(L_G_Y)
                R_G_X = self.extendFirstLast(R_G_X)
                R_G_Y = self.extendFirstLast(R_G_Y)
                #R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y
                MID_N_X = self.extendFirstLast(MID_N_X)
                MID_N_Y = self.extendFirstLast(MID_N_Y)

                L_G_X = self.interpolate(L_G_X)
                L_G_Y = self.interpolate(L_G_Y)
                R_G_X = self.interpolate(R_G_X)
                R_G_Y = self.interpolate(R_G_Y)
                #R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y
                MID_N_X = self.interpolate(MID_N_X)
                MID_N_Y = self.interpolate(MID_N_Y)
                # Needle Passing First
                #if(self.task == "Needle_Passing"):
                #    for i in range(1,trialLength):
                #        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i]);
                
                if(self.task == "Needle_Passing"):
                    R_4_X = self.extendFirstLast(R_4_X)
                    R_4_Y = self.extendFirstLast(R_4_Y)
                    R_5_X = self.extendFirstLast(R_5_X)
                    R_5_Y = self.extendFirstLast(R_5_Y)
                    R_6_X = self.extendFirstLast(R_6_X)
                    R_6_Y = self.extendFirstLast(R_6_Y)
                    R_7_X = self.extendFirstLast(R_7_X)
                    R_7_Y = self.extendFirstLast(R_7_Y)

                    R_4_X = self.interpolate(R_4_X)
                    R_4_Y = self.interpolate(R_4_Y)
                    R_5_X = self.interpolate(R_5_X)
                    R_5_Y = self.interpolate(R_5_Y)
                    R_6_X = self.interpolate(R_6_X)
                    R_6_Y = self.interpolate(R_6_Y)
                    R_7_X = self.interpolate(R_7_X)
                    R_7_Y = self.interpolate(R_7_Y)
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i])+","+str(R_4_X[i])+","+str(R_4_Y[i])+","+str(R_5_X[i])+","+str(R_5_Y[i])+","+str(R_6_X[i])+","+str(R_6_Y[i])+","+str(R_7_X[i])+","+str(R_7_Y[i]);
            
                elif(self.task=="Knot_Tying"):
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i]);
                else:
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i]);

            self.save(newFileName+".csv",trialKin)
            print("Saving",newFileName, "with", numAnnotations,"/",len(trialKin)," # ",str(numAnnotations/len(trialKin)),"labeled")


    def addVisData(self):
        all_kin_data = self.getKinLines();
        countTotal = 0;
        countMissing = 0;
        
        for trial in all_kin_data:
            trial_s = trial.replace(".csv","")
            thisTrialDir = os.path.join(self.kinLabelDir,trial_s)
            trialKin = all_kin_data[trial]
            newFileName = os.path.join(self.kinOutput,trial_s)
            
            if not os.path.exists(thisTrialDir):
                #os.makedirs(newFileDir)
                print("No Annotations for ",thisTrialDir)
                countMissing = countMissing+1;
                continue;
            else: 
                # make a list with 0 for no annotation
                # write self.interpolate(list) to the file
                numAnnotations = 0;
                trialLength = len(trialKin)
                pre_inter_labels = [];

                for i in range(trialLength):
                    #pre_inter_labels.append(trialKin[i]);
                    pre_inter_labels.append(0);
                #for i in range(trialLength):
                if(self.task=="Needle_Passing"):
                    trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y,Needle_X,Needle_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y"
                else: 
                    trialKin[0]+=",L_Gripper_X,L_Gripper_Y,R_Gripper_X,R_Gripper_Y,Needle_X,Needle_Y"
                #L_Gripper_X
                #L_Gripper_Y
                #R_Gripper_X
                #L_Gripper_Y
                L_G_X = pre_inter_labels.copy()
                L_G_Y = pre_inter_labels.copy()
                R_G_X = pre_inter_labels.copy()
                R_G_Y = pre_inter_labels.copy()

                MID_N_X = pre_inter_labels.copy()
                MID_N_Y = pre_inter_labels.copy()

                R_4_X = pre_inter_labels.copy()
                R_4_Y = pre_inter_labels.copy()
                R_5_X = pre_inter_labels.copy()
                R_5_Y = pre_inter_labels.copy()
                R_6_X = pre_inter_labels.copy()
                R_6_Y = pre_inter_labels.copy()
                R_7_X = pre_inter_labels.copy()
                R_7_Y = pre_inter_labels.copy()

                for root, dirs, files in os.walk(thisTrialDir):
                    for file in files:          
                        if "frame" not in file:
                            continue
                        f_n = file.replace(".json","").replace("frame_","")
                        f_n = int(f_n)
                        #trialKin[f_n] = trialKin[f_n]+",Value"

                        numAnnotations= numAnnotations+1
                        thisKinFileName = os.path.join(thisTrialDir,file)
                        #pre_inter_labels = self.addThisAFileToKin(thisKinFileName,trial,f_n,pre_inter_labels)
                        if(self.task == "Needle_Passing"):
                            L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y = self.generateAnnotationSequencesNP(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y,R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y)
                        else:
                            L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y = self.generateAnnotationSequences(thisKinFileName,trial,f_n,L_G_X, L_G_Y, R_G_X, R_G_Y,MID_N_X,MID_N_Y)
                        
                                
                    #print(trial)
            
                L_G_X = self.interpolate(L_G_X)
                L_G_Y = self.interpolate(L_G_Y)
                R_G_X = self.interpolate(R_G_X)
                R_G_Y = self.interpolate(R_G_Y)
                #R_4_X,R_4_Y,R_5_X,R_5_Y,R_6_X,R_6_Y,R_7_X,R_7_Y
                MID_N_X = self.interpolate(MID_N_X)
                MID_N_Y = self.interpolate(MID_N_Y)

                if(self.task == "Needle_Passing"):
                    R_4_X = self.interpolate(R_4_X)
                    R_4_Y = self.interpolate(R_4_Y)
                    R_5_X = self.interpolate(R_5_Y)
                    R_5_Y = self.interpolate(R_5_Y)
                    R_6_X = self.interpolate(R_6_Y)
                    R_6_Y = self.interpolate(R_6_Y)
                    R_7_X = self.interpolate(R_7_Y)
                    R_7_Y = self.interpolate(R_7_Y)
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i])+","+str(R_4_X[i])+","+str(R_4_Y[i])+","+str(R_5_X[i])+","+str(R_5_Y[i])+","+str(R_6_X[i])+","+str(R_6_Y[i])+","+str(R_7_X[i])+","+str(R_7_Y[i]);
            
                else: 
                    for i in range(1,trialLength):
                        trialKin[i] = trialKin[i]+","+str(L_G_X[i])+","+str(L_G_Y[i])+","+str(R_G_X[i])+","+str(R_G_Y[i])+","+str(MID_N_X[i])+","+str(MID_N_Y[i]);
            
            self.save(newFileName+".csv",trialKin)
            print("Saving",newFileName, "with", numAnnotations,"/",len(trialKin)," # ",str(numAnnotations/len(trialKin)),"labeled")
       
    def prepareImages(self):
        # I need to be able to have an array of LINES of kinematic data.
        all_kin_data = self.getKinLines();
        superCount = 0;
        for trial in all_kin_data:
            print("For Trial:",trial,len(all_kin_data[trial]))
            # I need to be able to LOOP though each annotation for a given TRIAL
            count = 0;
            
            for root, dirs, files in os.walk(self.taskDirButImages):
                for file in files:          
                    if "frame" not in file:
                        continue

                    lastRoot = os.path.basename(root)
                    d_subject, d_trial, d_trialStr, StartFrame, EndFrame = self.decode(str(lastRoot))

                    frameFilename = os.path.join(root,file)
                    newFileDir = os.path.join(self.imgLabelDir,self.getFormattedName(d_subject, d_trialStr))
                    newFilename = self.getFormattedName(d_subject, d_trialStr)+"_"+StartFrame+"_"+EndFrame+".png"
                    #filename = os.path.basename(item[0])                
                    #copyfile(item[0], os.path.join("/Users/username/Desktop/testPhotos", filename))
                    #print("copy:",frameFilename,newFileDir,newFilename)
                    #if not os.path.exists(newFileDir):
                    #    os.makedirs(newFileDir)
                    
                    #shutil.copyfile(frameFilename, fullName) 
                    T_S_T = self.getFormattedName(d_subject, d_trialStr) # of the annotations ex. 'Knot_Tying_S02_T01'
                    if(T_S_T== trial.replace(".csv","")):  

                        # OK, now you can replace frame_000X with frame_StartFrame+000X
                        # these frame_0123.png__objects.json will be an absolute order over the set of json files in a particular trial
                        # decompose file into frame_ and .json  
                        frame_n = int(self.getFrameNumberIMG(file))

                        realOffset = 0;
                        frameDict = self.frameOffset(lastRoot)
                        if frameDict is None: 
                            print("frame dict none for",newFileDir,"in",file,"root",lastRoot)
                        for itm in frameDict: # 'NoneType' object is not iterable
                            if(itm[1] == frame_n):
                                #print(T_S_T,"-",str(StartFrame),"-", self.getFrameNumber(file),"->real frame number is ",itm[0])
                                realOffset = itm[0]

                        f_n = int(realOffset) + int(StartFrame)
                        destName = "frame_"+self.pad4zeroes(f_n)
                        fullName = os.path.join(newFileDir, destName)
                        #print("\tSame T_S_T:",T_S_T,"-",str(StartFrame),"-",file,"to:",destName, " in dir full:",fullName)
                        
                        #for name, age in frameDict.items():
                        #    if age == self.getFrameNumber(file):
                        #        print(name)
                        #quit()                        
                        if not os.path.exists(newFileDir):
                            os.makedirs(newFileDir)
                        print("Saved:",frameFilename,"to:",fullName+".png")
                        #shutil.copyfile(frameFilename, fullName+".png") 
                        
                        
                        count += 1
                        #6:00 good job. Now you need to make a helper function to collect all of those definitons indexed by [the wholw ex KnotTying_B002_1200_1260]
                        # so then you can see its key. that is the actual frame number offset
                    # file belong to the right task
            print("there are",count, "counts")
            superCount = superCount+1;
            #return
        #print(self.labelDict)
        print("Total",superCount)

    def prepareAnnotations(self):
        # I need to be able to have an array of LINES of kinematic data.
        all_kin_data = self.getKinLines();
        superCount = 0;
        for trial in all_kin_data:
            print("For Trial:",trial,len(all_kin_data[trial]))
            # I need to be able to LOOP though each annotation for a given TRIAL
            count = 0;
            for root, dirs, files in os.walk(self.taskDir):
                for file in files:          
                    if "frame" not in file:
                        continue

                    lastRoot = os.path.basename(root)
                    d_subject, d_trial, d_trialStr, StartFrame, EndFrame = self.decode(str(lastRoot))

                    frameFilename = os.path.join(root,file)
                    newFileDir = os.path.join(self.kinLabelDir,self.getFormattedName(d_subject, d_trialStr))
                    newFilename = self.getFormattedName(d_subject, d_trialStr)+"_"+StartFrame+"_"+EndFrame+".json"
                    #filename = os.path.basename(item[0])                
                    #copyfile(item[0], os.path.join("/Users/username/Desktop/testPhotos", filename))
                    #print("copy:",frameFilename,newFileDir,newFilename)
                    #if not os.path.exists(newFileDir):
                    #    os.makedirs(newFileDir)
                    
                    #shutil.copyfile(frameFilename, fullName) 
                    T_S_T = self.getFormattedName(d_subject, d_trialStr) # of the annotations ex. 'Knot_Tying_S02_T01'
                    if(T_S_T== trial.replace(".csv","")):  

                        # OK, now you can replace frame_000X with frame_StartFrame+000X
                        # these frame_0123.png__objects.json will be an absolute order over the set of json files in a particular trial
                        # decompose file into frame_ and .json  
                        frame_n = int(self.getFrameNumber(file))

                        realOffset = 0;
                        frameDict = self.frameOffset(lastRoot)
                        if frameDict is None: 
                            print("frame dict none for",newFileDir,"in",file,"root",lastRoot)
                        for itm in frameDict: # 'NoneType' object is not iterable
                            if(itm[1] == frame_n):
                                #print(T_S_T,"-",str(StartFrame),"-", self.getFrameNumber(file),"->real frame number is ",itm[0])
                                realOffset = itm[0]

                        f_n = int(realOffset) + int(StartFrame)
                        destName = "frame_"+self.pad4zeroes(f_n)
                        fullName = os.path.join(newFileDir, destName)
                        #print("\tSame T_S_T:",T_S_T,"-",str(StartFrame),"-",file,"to:",destName, " in dir full:",fullName)
                        
                        #for name, age in frameDict.items():
                        #    if age == self.getFrameNumber(file):
                        #        print(name)
                        #quit()                        
                        if not os.path.exists(newFileDir):
                            os.makedirs(newFileDir)
                        shutil.copyfile(frameFilename, fullName+".json")                        
                        
                        count += 1
                        # Now you need to make a helper function to collect all of those definitons indexed by [the wholw ex KnotTying_B002_1200_1260]
                        # so then you can see its key. that is the actual frame number offset
                    # file belong to the right task
            print("there are",count, "counts")
            superCount = superCount+1;
            #return
        #print(self.labelDict)
        print("Total",superCount)
    
    def frameOffset(self,lastRoot):
        
        for gesture in self.RecordData.keys():
                #print("gesture:",gesture)
                for segmented in self.RecordData[gesture]:
                    if(segmented["name"][0:-2] in lastRoot):
                        #print(segmented["name"])
                        return segmented["frames"]

    def frameOffsetSlow(self,lastRoot):
        #self.json_location = jsonLoc    
        with open(self.recordsLoc) as f:
            data = json.load(f)
            for gesture in data.keys():
                #print("gesture:",gesture)
                for segmented in data[gesture]:
                    if(segmented["name"] in lastRoot):
                        #print(segmented["name"])
                        return segmented["frames"]
    
    def getKinLines(self):
        count = 0
        all_kin_data = {}
        for root, dirs, files in os.walk(self.kinSource):
            for file in files:
                if("99"  in file):
                    continue
                #print(file)
                kin_file =  os.path.join(self.kinSource, file)

                kin_lines = []
                with open(kin_file) as kin_data:
                    for line in kin_data:
                        kin_lines.append(line.strip())

                i = 0 
                for line in kin_lines:
                    line_ = line.replace("\n","")
                    i=i+1
                kin_len = i;
                #print("\tKin_Length: ", kin_len)
                count=count+1
                all_kin_data[file] = kin_lines         

        print(count,"files processed!")
        return all_kin_data

    def MatchFormattingImageLabel(self):
        count = 0;
        taskDir = os.path.join(self.imagesDir, self.task)
        for root, dirs, files in os.walk(taskDir):           
            for file in files:
                if "sub_Sut"  in root:
                    continue
                try:
                    print("root:",root,"file",file)
                    rest, sectionName = os.path.split(root) 
                    head, subName = os.path.split(rest)
                    
                    print("\tchange",subName+"@"+sectionName)
                    print("\tto:",os.path.join(head,subName+"_"+sectionName))
                    os.rename(root,os.path.join(head,subName+"_"+sectionName))
                except Exception as e:
                    print(e)
                #head, tail = os.path.split()
        
    def ProcImages(self):
        count = 0;
        for root, dirs, files in os.walk(self.taskDir):
            for file in files:
                if "frame" not in file:
                    continue
                imageRoot = root
                lastRoot = os.path.basename(root)
                subj, trial,trialStr, StartFrame, EndFrame = self.decode(str(lastRoot)) # G1_sub_NeedlePassing_B001_166_586
                #labelRoot = self.getDirectory(root,"labels")
                #outputRoot =  self.getDirectory(root,"output")

                if (subj in self.labelDict):
                    if(trial in self.labelDict[subj] ):
                        if(StartFrame  in self.labelDict[subj][trial]):
                            pass;
                        else: #
                            self.labelDict[subj][trial][StartFrame] = EndFrame;
                    else: 
                        # trial is NOT in the dict                        
                        self.labelDict[subj][trial] = {};
                        self.labelDict[subj][trial][StartFrame] = EndFrame;
                else:
                    self.labelDict[subj] = {}
                    self.labelDict[subj][trial] = {};
                    self.labelDict[subj][trial][StartFrame] = EndFrame;


                #print("File:",file, " Code",TestTrial, "Start",StartFrame,"End",EndFrame)
                imageSource = os.path.join(imageRoot, file)

                #labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                #outputDest = os.path.join(outputRoot, file)
                '''

                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)

                if os.path.exists(outputDest):
                    os.remove(outputDest)

                if not os.path.exists(labelSource):
                    print("label not found for ",imageSource)
                    continue
                else:
                    self.DrawLabel(imageSource,labelSource,outputDest)
                '''
                count += 1
        print(count,"files processed!")
        #print(self.labelDict)

    def decode(self, fold):
        f_l = fold.split("_")
        subj = f_l[-3][0]
        trial = f_l[-3][1:]
        trialStr = f_l[-3][2:]
        return subj, trial,trialStr,f_l[-2],f_l[-1];
        
    def getSubject(self, subject):
        if subject =="B":
            subject = 2
        elif subject == "C":
            subject = 3
        elif subject == "D":
            subject = 4
        elif subject == "E":
            subject = 5
        elif subject == "F":
            subject = 6
        elif subject == "G":
            subject = 7
        elif subject == "H":
            subject = 8
        elif subject == "I":
            subject = 9

        return subject

class Iterator:
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images_pre")
        self.labelsDir = os.path.join(self.CWD, task,"annotations_pre")
        self.outputDir = os.path.join(self.CWD, task,"labeled_images")
        #self.dirIndices = [len(self.CWD)+1, len(self.CWD) + len("output")+1]

    def count_and_add(self, className, names):
        global absolute_counts
        if(className in names): #'Needle Mask', 'Needle Mask', 'Needle Mask', 'Needle Mask', 'Left Grasper', 'Right Grasper']
            if(className in absolute_counts):
                absolute_counts[className] +=1
            else:
                absolute_counts[className] = 1

    def just_add(self, className, names):
        global state_counts    
        if(className in state_counts):
            state_counts[className] +=1
        else:
            state_counts[className] = 1

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

        font = ImageFont.truetype("arial.ttf", 12)
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

    def DrawSingleImage(self, imageSource, labelSource, target, DEBUG=False):
       
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        font = ImageFont.truetype("arial.ttf", 12)
        #print(polyLines)
        #return    
        img = Image.open(imageSource)
        #draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   

        self.DrawPolygons(polygons,polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)

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

    def DrawPolygons(self, polygons,polyNames,draw,font):
        for i in range(len(polygons)):
            if("Ring" in polyNames[i]):
                c = self.getRBGA(colors[i])
                #print("Poly1:",polygons[i])
                draw.polygon(polygons[i], fill=c) #,outline='#EA5536')     
                ########## CENTER POINT
                x_c, y_c = self.Centroid(polygons[i])          
                leftUpPoint = (x_c-radius, y_c-radius)
                rightDownPoint = (x_c+radius, y_c+radius)
                twoPointList = [leftUpPoint, rightDownPoint]
                c = self.getRBGA(colors[i+(len(polygons))])
                draw.ellipse(twoPointList, fill=c)            
                # draw.text((x, y),"Sample Text",(r,g,b))
                draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)


    def DrawLabel(self, imageSource, labelSource, target, debug=False):
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        global state_counts
        global absolute_counts
        absolute_counts["Labeled Frames"] +=1;
        for name in polyNames:
            self.count_and_add(name,polyNames)

        for name in kpNames:
            self.count_and_add(name, kpNames)

        for name in polyLineNames:
            self.count_and_add(name, polyLineNames)

        polyNames_s = Counter(polyNames)
        for name in polyNames_s:
            self.just_add(name+"_"+str(polyNames_s[name]), polyNames)
            #print(name,polyNames_s[name])

        kpNames_s = Counter(kpNames)
        for name in kpNames_s:
            self.just_add(name+"_"+str(kpNames_s[name]), kpNames)
            #print(name,kpNames_s[name])

        polyLineNames_s = Counter(polyLineNames)
        for name in polyLineNames_s:
            self.just_add(name+"_"+str(polyLineNames_s[name]), polyLineNames)
            #print(name,polyLineNames_s[name])


        font = ImageFont.truetype("arial.ttf", 12)
        #print(polyLines)
        #return    
        img = Image.open(imageSource)
        #draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   
        ''' no polys
        '''
        for i in range(len(polygons)):
            if("Ring" in polyNames[i]):
                c = self.getRBGA(colors[i])
                #print("Poly1:",polygons[i])
                draw.polygon(polygons[i], fill=c) #,outline='#EA5536')     
                ########## CENTER POINT
                x_c, y_c = self.Centroid(polygons[i])          
                leftUpPoint = (x_c-radius, y_c-radius)
                rightDownPoint = (x_c+radius, y_c+radius)
                twoPointList = [leftUpPoint, rightDownPoint]
                c = self.getRBGA(colors[i+(len(polygons))])
                draw.ellipse(twoPointList, fill=c)            
                # draw.text((x, y),"Sample Text",(r,g,b))
                draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)
       
        '''
        polyIndex = 0
        for poly in polyLines:
            currPoly_X = []
            currPoly_Y = []
            for i in range(0,len(poly),2):
                currPoly_X.append(poly[i])
                currPoly_Y.append(poly[i+1])
            #colors = np.random.rand()*10
            ax1.scatter(currPoly_X, currPoly_Y, s=10, c=np.random.rand(3,), marker="s", label="index:"+str(polyIndex))
            #ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
            polyIndex+=1
        '''
        
        
        
        thread_X = []
        thread_Y = []       
        for i in range(len(polyLines)):
            c = self.getRBGA(colors[-i])
            k=0
            for j in range(0,len(polyLines[i]),2):
                thread_X.append(polyLines[i][j])
                thread_Y.append(polyLines[i][j+1])

            for j in range(0,len(polyLines[i])):
                #if(k+3>=len(polyLines[i])):
                #    
                draw.line(( polyLines[i][k],
                            polyLines[i][k+1],
                            polyLines[i][k+2],
                            polyLines[i][k+3]), fill=c, width=9) 
                k+=2
                if(k>=len(polyLines[i])-2): break
                
            draw.text((polyLines[i][0],polyLines[i][1]),polyLineNames[i]+str(i),(255,255,255),font=font)     

        kk=0   
        t_min = min(len(thread_X),len(thread_Y))
        thread_X = thread_X[0:t_min]
        thread_Y = thread_Y[0:t_min]   
        if t_min!=0:
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
                #draw.text((x-radius*2, y-radius),kpNames[i]+str(i),(0,0,0),font=font)

        '''
        for i in range(len(polyLines)): # draws each polygon 
            #print("Drawing one polyLine")
            c = self.getRBGA(colors[-i])
            k=0
            for j in range(0,len(polyLines[i])):
                #if(k+3>=len(polyLines[i])):
                #    
                draw.line(( polyLines[i][k],
                            polyLines[i][k+1],
                            polyLines[i][k+2],
                            polyLines[i][k+3]), fill=c, width=5) 
                k+=2
                if(k>=len(polyLines[i])-2): break
                
            draw.text((polyLines[i][0],polyLines[i][1]),"Polyline"+str(i),(255,255,255),font=font)            
        '''
                
        for i in range(len(KeyPoint)): # draws each KeyPoint
            x = KeyPoint[i][0]
            y = KeyPoint[i][1]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList, fill=c)
            draw.text((x-radius*2, y-radius),kpNames[i]+str(i),(255,255,255),font=font)
        

        img.save(target) # to save

    def DrawLabels(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                #if "Suturing_S02_T05" not in os.path.basename(root):
                #    continue
                print("Proc:", os.path.basename(root),file )
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
                        self.DrawSingleImage(imageSource,labelSource,outputDest)

                count += 1
                
        print(count,"images processed!")
       

'''
Usage: put the top folder name. This folder should exist both under imges and labels

The organization within both folders should be the same.

The labels will appear in the same file organization as the images dataset under the folder "output"
'''

main();

