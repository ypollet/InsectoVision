import os
from src.models.coords import Coords
from src.consts import *
from wand.image import Image

#One image of an entomology box
#Instance created only when images are loaded into the viewer 
#(not when scanning with neural net) 
class EntoBox:

    def __init__(self,name,img_path,bboxes_path = None, conf_threshold : float = DEFAULT_CONF):
        self.name = name

        #Get the image
        self.image = img_path
        with Image() as img:
            t = img.ping(filename=f'{img_path}[0]')
            width, height = t.width, t.height

        print(f"Image {name} dim = {width}x{height}")

        #Get the bboxes
        self.bboxes = []
        self.conf_threshold = conf_threshold
        if bboxes_path != None:
            self.get_bboxes(bboxes_path, width, height)


    def get_bboxes(self,bboxes_path, width, height):
        if(os.path.isfile(os.path.join(bboxes_path,self.name+".txt"))):
            self.bboxes = []
            txt = open(os.path.join(bboxes_path,self.name+".txt"))

            #Compute bbox coordinates from yolo notation
            for line in txt:
                la = line.split(" ")[0:6]                                         
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                x1 = (x-w/2)*width
                x2 = (x+w/2)*width
                y1 = (y-h/2)*height
                y2 = (y+h/2)*height
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
        self.coord : Coords = Coords.from_coords(*coord)
        self.conf : float = conf
        self.is_selected = False

    def to_yolo(self):
        # TODO : add height and width
        x1, y1, x2, y2 = self.coord.to_list()

        x = ((float(x2+x1))/2)
        y = ((float(y2+y1))/2)
        w = float(abs(x2-x1))
        h = float(abs(y2-y1))

        return [x,y,w,h]

    def update_status(self,ct):
        if self.status in [None,DOUBT,SURE]:
            if(self.conf < ct):
                self.status = DOUBT
            else:
                self.status = SURE

