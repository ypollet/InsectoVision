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

    def __init__(self,name,img_path,ai_yolo_path = None, saved_path = None, conf_threshold : float = DEFAULT_CONF):
        self.name = name

        #Get the image
        self.image = img_path
        if img_path != None:
            self.image = img_path
            with Image() as img:
                t = img.ping(filename=f'{img_path}[0]')
                self.width, self.height = t.width, t.height
        else:
            self.width = 1
            self.height = 1

        
        self.saved = BooleanVar(value=saved_path != None)

        #Get the bboxes
        self.bboxes = []
        self.groups = []
        self.conf_threshold = conf_threshold
        self.ai_labels = ai_yolo_path
        self.saved_labels = saved_path
        if self.ai_labels != None or self.saved_labels != None:
            self.load_bboxes()

    def load_bboxes(self):
        print(f"Loading bboxes for {self.name} : {self.saved_labels} / {self.ai_labels}")
        try:
            if(self.saved_labels != None):
                if self.saved_labels.endswith(".json"):
                    self.bboxes, self.groups, self.conf_threshold = read_bbox_json(self.saved_labels, self)
                    return
                if self.saved_labels.endswith(".txt"):
                    self.bboxes = read_yolo(self.saved_labels, self)
                    return
        except ValueError:
            pass
        try:
            if self.ai_labels != None and self.ai_labels.endswith(".txt"):
                self.bboxes = read_yolo(self.ai_labels, self)
        except ValueError:
            pass
            

    def save(self, saved_path):
        self.saved.set(True)
        self.saved_path = saved_path

    def is_saved(self):
        return self.saved_path != None

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

def read_bbox_json(json_path, parent : EntoBox):
    if not os.path.exists(json_path) or not json_path.endswith(".json"):
        raise ValueError("invalid path")
    bboxes = []
    groups = []
    with open(json_path, "r") as f:
        bboxes_dict = json.load(f)
            
    try :
        conf_threshold = bboxes_dict["conf"]
    except:
        conf_threshold = DEFAULT_CONF
            
    for bbox in bboxes_dict["bboxes"]:
                
        [x,y,w,h] = bbox["position"]
        x1 = (x-w/2)*bboxes_dict["width"]
        x2 = (x+w/2)*bboxes_dict["width"]
        y1 = (y-h/2)*bboxes_dict["height"]
        y2 = (y+h/2)*bboxes_dict["height"]

        bboxes.append(BBox([x1,y1,x2,y2],parent=parent, conf=bbox["conf"], group=bbox["group"], label=bbox["label"]))
                
        if bbox["group"] not in groups:
            groups.append(bbox["group"])
    return bboxes, groups, conf_threshold

def read_yolo(txt_path, parent : EntoBox):
    if not os.path.exists(txt_path) or not txt_path.endswith(".txt"):
        raise ValueError("invalid path")
    bboxes = []
    txt = open(txt_path)

    #Compute bbox coordinates from yolo notation
    for line in txt:
        la = line.split(" ")[0:6]                                         
        [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
        x1 = (x-w/2)*parent.width
        x2 = (x+w/2)*parent.width
        y1 = (y-h/2)*parent.height
        y2 = (y+h/2)*parent.height
        if(len(la) == 6):
            c = float(la[5])
        else:
            c = 1
        bboxes.append(BBox([x1,y1,x2,y2],c,parent=parent))
    txt.close()
    return bboxes