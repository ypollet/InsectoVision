import os
import json
from src.models.coords import Coords
from src.consts import *
from wand.image import Image

from tkinter import BooleanVar

#One image of an entomology box
#Instance created only when images are loaded into the viewer 
#(not when scanning with neural net) 
class EntoBox:

    def __init__(self,name,img_path,bboxes_path = None, conf_threshold : float = DEFAULT_CONF, is_saved = None):
        self.name = name

        #Get the image
        self.image = img_path
        with Image() as img:
            t = img.ping(filename=f'{img_path}[0]')
            self.width, self.height = t.width, t.height

        
        self.saved = BooleanVar(value=is_saved or False)

        #Get the bboxes
        self.bboxes = []
        self.groups = []
        self.conf_threshold = conf_threshold
        if bboxes_path != None:
            self.get_bboxes(bboxes_path)

    def get_bboxes(self,bboxes_path):
        if(os.path.isfile(os.path.join(bboxes_path,self.name+".json"))):
            with open(os.path.join(bboxes_path,self.name+".json"), "r") as f:
                bboxes_dict = json.load(f)
            
            try :
                self.conf_threshold = bboxes_dict["conf"]
            except:
                self.conf_threshold = DEFAULT_CONF
            
            for bbox in bboxes_dict["bboxes"]:
                
                [x,y,w,h] = bbox["position"]
                x1 = (x-w/2)*self.width
                x2 = (x+w/2)*self.width
                y1 = (y-h/2)*self.height
                y2 = (y+h/2)*self.height

                self.bboxes.append(BBox([x1,y1,x2,y2],parent=self, conf=bbox["conf"], group=bbox["group"], label=bbox["label"]))
                
                if bbox["group"] not in self.groups:
                    self.groups.append(bbox["group"])
            return
        if(os.path.isfile(os.path.join(bboxes_path,self.name+".txt"))):
            self.bboxes = []
            txt = open(os.path.join(bboxes_path,self.name+".txt"))

            #Compute bbox coordinates from yolo notation
            for line in txt:
                la = line.split(" ")[0:6]                                         
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                x1 = (x-w/2)*self.width
                x2 = (x+w/2)*self.width
                y1 = (y-h/2)*self.height
                y2 = (y+h/2)*self.height
                if(len(la) == 6):
                    c = float(la[5])
                else:
                    c = 1
                self.bboxes.append(BBox([x1,y1,x2,y2],c,self))
            txt.close()

    def set_saved(self, value):
        self.saved.set(bool(value))

    def is_saved(self):
        return self.saved.get()

class BBox:

    def __init__(self,coord,conf : float,parent : EntoBox, label : str = DEFAULT_LABEL, group : str = ""):
        self.parent : EntoBox = parent

        self.itemId = None
        self.label = label
        self.group = group
        
        self.coord : Coords = Coords.from_coords(*coord)
        self.conf : float = conf
        self.is_selected = False

    def status(self):
        if self.is_selected:
            return Status.SELECTED 
        return self.conf_status()
    
    def conf_status(self):
        if self.conf == 1:
            return Status.CONFIRMED
        if self.conf == 0:
            return Status.REJECTED
        return Status.SURE if (self.conf >= self.parent.conf_threshold) else Status.DOUBT
    

    def color(self):
        return COLORS[self.status()]


    def to_yolo(self, width, height):
        x1, y1, x2, y2 = self.coord.to_list()

        x = ((float(x2+x1))/2) / width
        y = ((float(y2+y1))/2) / height
        w = float(abs(x2-x1))/ width
        h = float(abs(y2-y1))/ height

        return [x,y,w,h]

