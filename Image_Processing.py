'''
6/9/22 1:00
Date	MTML_position_x	MTML_position_y	MTML_position_z	MTML_orientation_x	MTML_orientation_y	MTML_orientation_z	MTML_orientation_w	MTML_gripper_angle	MTMR_position_x	MTMR_position_y	MTMR_position_z	MTMR_orientation_x	MTMR_orientation_y	MTMR_orientation_z	MTMR_orientation_w	MTMR_gripper_angle	PSML_position_x	PSML_position_y	PSML_position_z	PSML_orientation_x	PSML_orientation_y	PSML_orientation_z	PSML_orientation_w	PSML_gripper_angle	PSMR_position_x	PSMR_position_y	PSMR_position_z	PSMR_orientation_x	PSMR_orientation_y	PSMR_orientation_z	PSMR_orientation_w	PSMR_gripper_angle
'''
import os, sys
import json
import pathlib
import math
from turtle import position
from xmlrpc.client import Boolean
from cv2 import KeyPoint, threshold
from matplotlib.colors import cnames
import numpy as np
from PIL import Image, ImageDraw, ImageColor,ImageFont
from scipy import interpolate
import time
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.geometry import LineString
from dataclasses import dataclass
from itertools import accumulate

global MAX_LEN
MAX_LEN = 200

global Gripperthreshold
Gripperthreshold = -0.8

#mathematica colors no longer used
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
opacity = (255,)
# radius of keypoint
radius = 3
RGB = {
    "BG":(0,0,0), 

    # Polygons: the keys match exactly with elements in the list polyNmes
    "Left Grasper":(250,0,0),
    "Right Grasper":(0,250,0),
    "Ring_4":(0,0,250),
    "Ring_5":(0,0,250),
    "Ring_6":(0,0,250),
    "Ring_7":(0,0,250),
    "Needle Mask":(0,0,250),

    # Keypoints: the keys match with elements in list kpNames
    "Needle End":(0,0,250),
    "Needle Tip":(0,0,250),

    # PolyLines: the keys match with elements in list polyLineNames
    "Thread Polyline":(0,0,250),
    "Bottom Left Thread":(0,0,250),
    "Bottom Right Thread":(0,0,250),
    "Top Left Thread":(0,0,250),
    "Top Right Thread":(0,0,250),
    
}
#BG = (0,0,0)
#GRASPER = (255,0,0)


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
    task = "Needle_Passing"
    I = ImageProcessor(task)
    I.DrawLabels()
    quit();    
   
class ImageProcessor:
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")
        self.labelsDir = os.path.join(self.CWD, task,"annotations")
        self.outputDir = os.path.join(self.CWD, task,"labeled_images")        
        self.mask_output = os.path.join(self.CWD,task,"mask_output")
        self.mpDir = os.path.join(self.CWD, task, "motion_primitives_combined")        
        self.mpDir_R = os.path.join(self.CWD, task, "motion_primitives_R")        
        self.mpDir_L = os.path.join(self.CWD, task, "motion_primitives_L")
        self.ContextDir = os.path.join(self.CWD,task,"transcriptions")

        self.OS = "windows"
    
    #! Needs: self.imagesDir, 
    def DrawLabels(self):
        count = 0
        numpyArr = []
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                print("Processing:", os.path.basename(root),file+".txt" )
                
                Bname = os.path.basename(root)

                '''
                If we replace "images" by "labels" then the image source should be the same as the label source, 
                which is the same as the output destination
                '''
                imageRoot = root
                labelRoot = root.replace("images","annotations")
                outputRoot = root.replace("images","mask_output")

                imageSource = os.path.join(imageRoot, file)
                labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                outputDest = os.path.join(outputRoot, file)

                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)
                if self.OS == "windows":
                    font = ImageFont.truetype("arial.ttf", 12)
                else: 
                    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")

                if not os.path.exists(labelSource):
                    #print("label not found for ",imageSource)
                    continue
                
                else:
                    #self.DrawLabel(imageSource,labelSource,outputDest)
                    if("Knot" in self.task):
                        self.DrawSingleImageKT(imageSource,labelSource,outputDest,font)
                    else:
                        self.DrawSingleImage(imageSource,labelSource,outputDest,font)

                count += 1                
        print(count,"images processed!")

    ''' Iterator: DrawSingleImageKT(imageSource, labelSource, target, DEBUG=False)
    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    :param DEBUG:       boolean - enable console printout

    :return: nothing'''     
    def DrawSingleImageKT(self, imageSource, labelSource, target, font,DEBUG=False):        
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); #! graspers only in KT, 
        polyLineNames, polyLines = J.getPolyLines();

        #! print(polyLines)
        #! return    
        img = Image.open(imageSource)
        #! draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   
        '''        
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
        '''
        self.flush_BG(img,draw)
        self.DrawPolygons(polygons,polyNames,draw,font)

        img.save(target)
        return 
            
    def flush_BG(self,img,draw):
        width,height=img.size
        draw.polygon([(0,0),(width,0),(width,height),(0,height)],RGB["BG"])
        #print("props",width,height)
        pass
    
    def DrawSingleImage(self, imageSource, labelSource, target, font, DEBUG=False):
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons();
        kpNames, KeyPoint = J.getKeyPoints(); 
        polyLineNames, polyLines = J.getPolyLines();
        

        img = Image.open(imageSource)
        IDX = int(imageSource.split("_")[-1].split(".")[0])
        #this is the image we return
        draw = ImageDraw.Draw(img, "RGBA")  

        #black background
        self.flush_BG(img,draw)

        self.DrawPolygons(polygons, polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)

        #distances = self.CalcDistances(polygons,polyNames,polyLines,polyLineNames)
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
        return


    def imageToJSON(self, file):
        fileArr = file.split(".")
        return "".join(fileArr[:-1]) + ".json"

    def getRBGA(self, hexColor):
        c = ImageColor.getcolor(hexColor, "RGB")        
        c = c + opacity
        return c

    def getRGBA_from_name(self, annotation_instance_name):
        if( annotation_instance_name in RGB.keys()):
            c = RGB[annotation_instance_name]
            c = c + opacity
            return c
        else:
            print("No color found for annotation class", annotation_instance_name)
        
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

    
    ''' Iterator: CalcDistancesSingleThread(self,LGX,LGY,RGX,RGY,ThreadX,ThreadY)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    MPI, CtxI,CtxI_Pred,
    :param DEBUG:       boolean - enable console printout

    :do: Draw a single image containing
        - ground truth context labels
        - predicted context labels, 
        - annotation objects (gripper mask, thread mask)

        using the video frame as background
    LG_Info, RG_Info, LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextKT(imageSource,labelSource,outputDest, MPI, CtxI,CtxI_Pred)
    #! used in DrawSingleImageContextKT  
    :return: '''        
    def CalcDistancesSingleThread(self,LGX,LGY,RGX,RGY,ThreadX,ThreadY):
        LG_Points = []
        RG_Points = []
        Thread_Points = []
        for i in range(len(LGX)):
            LG_Points.append(  (LGX[i],LGY[i]) )
        for i in range(len(RGX)):
            RG_Points.append(  (RGX[i], RGY[i])   )
        for i in range(len(ThreadX)):
            Thread_Points.append(  (ThreadX[i], ThreadY[i])   )

        LG = Polygon(LG_Points)
        RG = Polygon(RG_Points)
        Thread = LineString(Thread_Points)

        LG_Info = []
        try:
            LG_Info = [ LG.distance(Thread), LG.intersects(Thread) ]
        except Exception as e:
            #print(e)
            LG_Info = [e,""]

        RG_Info = []
        try:
            RG_Info = [ RG.distance(Thread), RG.intersects(Thread) ]
        except Exception as e:
            #print(e)
            RG_Info = [e,""]
        
        return LG_Info,RG_Info

    def CalcDistances(self, LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY):
        LG_Points = []
        RG_Points = []
        N_Points = []
        RingPoints = [[]]

        for i in range(len(LGX)):
            LG_Points.append(  (LGX[i],LGY[i]) )
        for i in range(len(RGX)):
            RG_Points.append(  (RGX[i], RGY[i])   )
        for i in range(len(NX)):
            N_Points.append(    (NX[i],NY[i]))
        

        for j in range(len(RingsX)):
            RingPoints.append([])
            for i in range(len(RingsX[j])):
                RingPoints[j].append(   (RingsX[j][i],RingsY[j][i])  )



        LG = Polygon(LG_Points)
        RG = Polygon(RG_Points)
        Needle = Polygon(N_Points)

        RingsArr = []
        for j in range(len(RingsX)):
            RingsArr.append( Polygon(RingPoints[j]))

        LG_Info = []
        try:
            LG_Info = [ LG.distance(Needle), LG.intersects(Needle) ]
        except Exception as e:
            #print(e)
            LG_Info = [e,""]
        RG_Info = []
        try:
            RG_Info = [ RG.distance(Needle), RG.intersects(Needle) ]
        except Exception as e:
            #print(e)
            RG_Info = [e,""]
        #N_Info = ["None","None"]

        intersections = False
        interIDX = -1
        d = []
        for j in range(len(RingsX)):
            try:
                if( Needle.intersects(RingsArr[j])):
                    intersections = True
                    interIDX = j
                d.append(Needle.distance(RingsArr[j]))
            except:
                pass
        
            

        if(intersections):
            N_Info = ["Intersecting Ring_"+str(interIDX+4),";"]
        else:
            #N_Info = ["No intersections", " ".join([str(s) for s in d])]
            index_min = np.argmin(d)
            N_Info = ["", "Needle is d="+str(d[index_min])+" to Ring_"+str(index_min+4) ]
        '''

        try:
            # I have 4 rings
            # let's see if there are any intersections:
            d = []
            intersections = False
            for j in range(len(RingsX)):
                if( Needle.intersects(RingsArr[j])):
                    Ints = True
                d.append(Needle.distance(RingsArr[j]))

            index_min = np.argmin(d)
            N_Info[0] = "No Inters"
            N_Info[1] = " d="+str(d[index_min])+" to Ring_"+str(index_min+4)

            if(intersections):
                for k in range(len(RingsX)):
                    if( Needle.intersects(RingsArr[k])):
                        N_Info[0] = "Inter Ring_"+str(k+4)
                        N_Info[1] = " d=0 to Ring_"+str(k+4)
                           
        except Exception as e:
            print(e)
        '''
        distances = d
        return LG_Info, RG_Info, N_Info, (interIDX+4), distances
        #return (LG.distance(Needle), LG.intersects(Needle)), (RG.distance(Needle), RG.intersects(Needle)), "rings"
   
    def OrganizePoints(self, polygons,polyNames):
        LGX = []
        LGY = []
        RGX = []
        RGY = []
        NX = []
        NY = []
        RingsX = [[],[],[],[]]
        RingsY = [[],[],[],[]]
        for i in range(len(polyNames)):
            if("Left Grasper" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    LGX.append(polygons[i][k])
                    LGY.append(polygons[i][k+1])
            elif("Right Grasper" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RGX.append(polygons[i][k])
                    RGY.append(polygons[i][k+1])
            elif("Needle Mask" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    NX.append(polygons[i][k])
                    NY.append(polygons[i][k+1])
            elif("Ring_4" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[0].append(polygons[i][k])
                    RingsY[0].append(polygons[i][k+1])            
            elif("Ring_5" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[1].append(polygons[i][k])
                    RingsY[1].append(polygons[i][k+1])
            elif("Ring_6" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[2].append(polygons[i][k])
                    RingsY[2].append(polygons[i][k+1])
            elif("Ring_7" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[3].append(polygons[i][k])
                    RingsY[3].append(polygons[i][k+1])
            else:
                print("Unknown Polygon Class",polyNames[i])
        return LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY

    def RenderThread_Arr(self, thread_X, thread_Y, draw, font):        
        kk=0   
        t_min = min(len(thread_X),len(thread_Y))
        thread_X = thread_X[0:t_min]
        thread_Y = thread_Y[0:t_min]       
        distance = np.cumsum(np.sqrt( np.ediff1d(thread_X, to_begin=0)**2 + np.ediff1d(thread_Y, to_begin=0)**2 ))
        distance = distance/distance[-1]        
        fx, fy = interp1d( distance, thread_X ), interp1d( distance, thread_Y )                
        alpha = np.linspace(0, 1, 10)
        alpha_context = np.linspace(0, 1, 100)
        x_regular, y_regular = fx(alpha), fy(alpha)
        x_detailed, y_detailed = fx(alpha_context), fy(alpha_context)
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
        return x_detailed, y_detailed
    
    def dist(self, A, B):
        return math.sqrt( (A[0] - B[0] )**2+( A[1]-B[1])**2 )

    def distNeedle(self, ax, ay,needleEnd ):
        return math.sqrt( (ax - needleEnd[0] )**2+( ay - needleEnd[1] )**2 )

    #! PutTheadInOrder(polyLines arr,needleEnd as keypoint)
    def PutTheadInOrder(self,polyLines_M,kp):
        thread_X = []
        thread_Y = []  
        polyLines = polyLines_M.copy()
        while(len(polyLines) > 0):    
            index_of_closest_thread_arr = 0; 
            position_of_closest = 0; #!  0 for "cannonical" - same position , 1 for end
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

            min_0 = min(thread_idx_0_distances) #! min of all distances between last end
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

    def DrawThreadKT(self,polyLines, polyLineNames, needleEnd, draw, font):  
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

        ThreadX, ThreadY = self.RenderThread_Arr(thread_X, thread_Y, draw, font)
        return ThreadX, ThreadY  

    def DrawThread(self,polyLines, polyLineNames, needleEnd, draw, font,text=False,interpolation=False, Draw_thread_points=False):  
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
            c = self.getRGBA_from_name(polyLineNames[i])
            #c = self.getRBGA(colors[-i])
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
                if Draw_thread_points:
                    draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                x = polyLines[i][k+2]
                y = polyLines[i][k+3]     
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                if Draw_thread_points:
                    draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                k+=2
                if(k>=len(polyLines[i])-2): break
            if text:
                draw.text((polyLines[i][0],polyLines[i][1]),polyLineNames[i]+str(i),(255,255,255),font=font)   
        if interpolation:
            ThreadX, ThreadY = self.RenderThread_Arr(thread_X, thread_Y, draw, font)
            return ThreadX, ThreadY
        else: return 0,0
            
    def DrawKeyPoints(self,KeyPoint, kpNames, polygons,draw,font,text=False):
        for i in range(len(KeyPoint)): # draws each KeyPoint
            x = KeyPoint[i][0]
            y = KeyPoint[i][1]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            #c = self.getRBGA(colors[i+(len(polygons))])
            c = self.getRGBA_from_name(kpNames[i])
            draw.ellipse(twoPointList, fill=c)
            if text:
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

    def DrawPolygons(self, polygons,polyNames,draw,font,text=False,center=False):
        for i in range(len(polygons)):
            #c = self.getRBGA(colors[i])
            c = self.getRGBA_from_name(polyNames[i])
            draw.polygon(polygons[i], fill=c)
                ########## CENTER POINT
            if center:
                x_c, y_c = self.Centroid(polygons[i])          
                leftUpPoint = (x_c-radius, y_c-radius)
                rightDownPoint = (x_c+radius, y_c+radius)
                twoPointList = [leftUpPoint, rightDownPoint]
                c = self.getRBGA(colors[i+(len(polygons))])
                draw.ellipse(twoPointList,outline=1, fill=c)            
            if text: 
                draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)

    def DrawLabelsContext(self):
            count = 0
            for root, dirs, files in os.walk(self.imagesDir):
                for file in files:
                    if "frame" not in file:
                        continue
                    #if "frame_1599" in file or "frame_1264" in file or "frame_0805" in file or "frame_1572" in file:
                    #    continue
                    #if "Suturing_S02_T05" not in os.path.basename(root):
                    #    continue
                    print("Proc:", os.path.basename(root),file+".txt" )
                    Bname = os.path.basename(root)

                    MP_comb = os.path.join(self.mpDir,Bname+".txt")
                    #print(MP_comb)
                    MPI = MPInterface(MP_comb) # turn on for MPs as well

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
                            LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContext(imageSource,labelSource,outputDest, MPI, CtxI)
                    
                    

                    # we can then use the object Frame to determine the context
                    # Contact/Hold Context:
                    # "Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"
                    #           0                   1         2         3               4       5       6


                    # Needle State in Suturing:
                    # "Not Touching", "Touching", "In"
                    #               0       1       2
                    # Frame# {0,2,3}, {0,2,3}, {0,2,3}, {0,2,3}, {0}

                    # Needle States in Needle Passing
                    # "Out of", "Touching","In"
                    #       0           1     2
                    # Frame# {0,2,3,5},{0,2,3,5},{0,2,3,5},{0,2,3,5},{0,1,2}

                    # Knot States in Knot Tying:
                    

                    count += 1
                    
            print(count,"images processed!")
    
    def save(self, file, lines):
        with open(file, 'w+') as f:
            for item in lines:
                f.write("%s\n" % item)

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
        self.empty = len(self.transcript) == 0

    def getContext(self,index):
        if(self.empty):
             return "Building Prediction"
        for i in range(1,len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if(int(l_s[0]) > index):
                #return " ".join(min(0,i-1))
                return self.transcript[max(0,i-1)]
        

main();



