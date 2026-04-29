import os
from src.models.coords import Coords
from src.consts import *

#One image of an entomology box
#Instance created only when images are loaded into the viewer 
#(not when scanning with neural net) 
class EntoBox:

    def __init__(self,name,img_path,bboxes_path = None):
        self.name = name
        #print("Loading "+name)

        #Get the image
        self.image = img_path

        #Get the bboxes
        self.bboxes = []
        if bboxes_path != None:
            self.get_bboxes(bboxes_path)


    def get_bboxes(self,bboxes_path):
        if(os.path.isfile(os.path.join(bboxes_path,self.name+".txt"))):
            self.bboxes = []
            txt = open(os.path.join(bboxes_path,self.name+".txt"))

            #Compute bbox coordinates from yolo notation
            for line in txt:
                la = line.split(" ")[0:6]                                         
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                x1 = int((x-w/2))
                x2 = int((x+w/2))
                y1 = int((y-h/2))
                y2 = int((y+h/2))
                if(len(la) == 6):
                    c = float(la[5])
                else:
                    c = 1
                self.bboxes.append(BBox([x1,y1,x2,y2],c,self))
            txt.close()

class BBox:

    status = DOUBT
    itemId = None
    label = DEFAULT_LABEL

    def __init__(self,coord,conf : float,parent : EntoBox):
        self.parent : EntoBox = parent
        self.coord : Coords = Coords(*coord)
        self.conf : float = conf
    
    def draw(self,gui):      
        self.update_status(gui.conf_threshold)
        boxid = gui.canvas.create_rectangle(self.coord.x1,self.coord.y1,self.coord.x2,self.coord.y2,outline=COLORS[self.status],width=2,tags=["bbox"])
        self.itemId = boxid
        gui.drawn_bboxes.append(self)
    
    def redraw(self,gui):
        gui.canvas.delete(self.itemId)
        self.draw(gui)

    def to_yolo(self):
        x = ((float(self.coord.x2+self.coord.x1))/2)
        y = ((float(self.coord.y2+self.coord.y1))/2)
        w = float(abs(self.coord.x2-self.coord.x1))
        h = float(abs(self.coord.y2-self.coord.y1))

        return [x,y,w,h]

    def update_status(self,ct):
        if self.status in [None,DOUBT,SURE]:
            if(self.conf < ct):
                self.status = DOUBT
            else:
                self.status = SURE

