import tkinter as tk

from tkinter import ttk

from src.components.canvas import CanvasImage
from src.models.boxes import BBox, EntoBox

from src.models.coords import Coords, Point
from src.consts import *

class EntoboxCanvas(CanvasImage):

    def __init__(self, placeholder: ttk.Frame, entobox : EntoBox):
        super().__init__(placeholder, entobox.image)
        self.entobox = entobox
        
        self.canvas.bind('<Button-1>',   self.__create_rec)
        self.canvas.bind('<B1-Motion>',   self.on_move_M1_held)
        self.canvas.bind('<ButtonRelease-1>',   self.on_M1_release) 

        self.bboxes_id : dict[int | str, BBox] = dict()
        self.points_id : dict[int | str, Point] = dict()

        self.selected = []
        self.moved = False
        self.rec_start = None # Drawing rectangle
        self.rec_drawn = None
        self.set_to_selecting()

    
    def __ratio2image(self, x, y):
        box_image = self.canvas.coords(self.container)  # get image area
        x = (x - box_image[0]) / self.imscale
        y = (y - box_image[1]) / self.imscale
        return x,y
    
    def __ratio2canvas(self, x, y):
        box_image = self.canvas.coords(self.container)  # get image area
        x = x * self.imscale + box_image[0]
        y = y * self.imscale + box_image[1]
        return x,y

    def set_to_selecting(self):
        self.selecting = True
        self.canvas.config(cursor="arrow")
        self.canvas.bind('<Control-ButtonRelease-1>',   self.add_to_selection)
        
    def set_to_drawing(self):
        self.selecting = False
        self.canvas.config(cursor="tcross")
        self.canvas.unbind('<Control-ButtonRelease-1>')
        

    def __create_rec(self,e : tk.Event):
        if e.widget.tag_click:
            return
        
        x, y = self.__ratio2image(self.canvas.canvasx(e.x), self.canvas.canvasy(e.y))
        self.rec_start = (x, y)
        self.rec_drawn = self.canvas.create_rectangle(x,y,x,y,outline=COLORS[Status.SELECTED],width=WIDTH_LINE)

    def on_move_M1_held(self,e):
        if e.widget.tag_click:
            return

        x1, y1 = self.__ratio2canvas(self.rec_start[0], self.rec_start[1])
        x2, y2 = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)

        self.canvas.coords(self.rec_drawn, x1, y1, x2, y2)

    def add_to_selection(self,e):
        if e.widget.tag_click:
            e.widget.tag_click = False
            return       
        self.canvas.delete(self.rec_drawn)
        self.rec_drawn = False

        x1, y1 = self.__ratio2canvas(self.rec_start[0], self.rec_start[1])
        x2, y2 = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        
        self.select_boxes_zone(x1,y1,x2,y2)
    
    def on_M1_release(self,e):
        if e.widget.tag_click:
            e.widget.tag_click = False
            return
        self.canvas.delete(self.rec_drawn)
        self.rec_drawn = False

        # get canvas coordinates
        x1, y1 = self.__ratio2canvas(self.rec_start[0], self.rec_start[1])
        x2, y2 = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)

        if self.selecting:
            self.unselect_all()
            self.select_boxes_zone(x1,y1,x2,y2)
        else:
            # update to image_coordinates
            x1, y1 = self.__ratio2image(x1, y1)
            x2, y2 = self.__ratio2image(x2, y2)
            self.new_bbox(x1,y1,x2,y2)

    def new_bbox(self, x1,y1,x2,y2):
        new = BBox([x1,y1,x2,y2],1,self.entobox)
                    
        self.entobox.bboxes.append(new)
        self.draw_bbox(new)
        self.canvas.event_generate("<<OnBBoxModified>>")

    def select_boxes_zone(self, x1, y1, x2, y2):
        selected_boxes = self.canvas.find_overlapping(x1, y1, x2, y2)
        for boxid in selected_boxes:
            bbox = self.bboxes_id.get(boxid)
            if bbox is not None and not bbox.is_selected:
                bbox.is_selected = True
                self.selected.append(boxid)
                self.update_bbox_color(bbox)
    def draw_all_bboxes(self):
        for bbox in self.entobox.bboxes:
            self.draw_bbox(bbox)

    def draw_bbox(self, bbox : BBox):
        if bbox.itemId is not None or bbox.status() == Status.REJECTED:
            return
        x1, y1 = bbox.coord.first.to_list()
        x2, y2 = bbox.coord.second.to_list()

        x1, y1 = self.__ratio2canvas(x1, y1)
        x2, y2 = self.__ratio2canvas(x2, y2)
        boxid = self.canvas.create_rectangle(x1,y1,x2,y2, outline=bbox.color(),width=WIDTH_LINE,tags=["bbox"]) #, fill=bbox.color(), stipple="gray12")
        self.canvas.tag_bind(boxid, '<Button-1>', lambda e: self.select_rec(boxid, e))
        self.canvas.tag_bind(boxid, '<Control-1>', lambda e: self.select_many_rec(boxid, e))
        
        bbox.itemId = boxid
        self.bboxes_id[boxid] = bbox

        point_id_first = self.canvas.create_oval(x1, y1,
                                    x1, y1,
                                    width=WIDTH_LINE * RADIUS_CIRCLE, outline=bbox.color())
        self.bind_points_events(point_id_first, boxid)
        bbox.coord.first.itemId = point_id_first
        self.points_id[point_id_first] = bbox.coord.first

        point_id_second = self.canvas.create_oval(x2, y2,
                                    x2, y2,
                                    width=WIDTH_LINE * RADIUS_CIRCLE, outline=bbox.color())
        self.bind_points_events(point_id_second, boxid)
        bbox.coord.second.itemId = point_id_second
        self.points_id[point_id_second] = bbox.coord.second

    def bind_points_events(self, pointId, boxId):
        self.canvas.tag_bind(pointId, '<B1-Motion>', lambda e: self.__move_point(pointId, boxId, e))
        self.canvas.tag_bind(pointId, '<ButtonRelease-1>', lambda e: self.confirm(boxId))
        self.canvas.tag_bind(pointId, '<Button-1>', lambda e: self.select_rec(boxId, e))
        self.canvas.tag_bind(pointId, '<Control-1>', lambda e: self.select_many_rec(boxId, e))

    def unselect_all(self):
        for boxid in self.selected:
            bbox = self.bboxes_id[boxid]
            bbox.is_selected = False
            self.update_bbox_color(bbox)
        
        self.selected = []

    def select_rec(self, boxid, event : tk.Event):
        if self.selecting:
            event.widget.tag_click = True
            self.unselect_all()
            self.select_many_rec(boxid, event)
            
    def select_many_rec(self, boxid, event):
        if self.selecting:
            event.widget.tag_click = True
            bbox = self.bboxes_id[boxid]
            bbox.is_selected = not bbox.is_selected
            if bbox.is_selected:
                self.selected.append(boxid)
            else:
                self.selected.remove(boxid)
            self.update_bbox_color(bbox)
        
    def combine_select_bboxes(self):
        if len(self.selected)<2:
            return
        

        coord = [float("inf"),float("inf"),float("-inf"),float("-inf")]
        for bbox_id in self.selected:
            bbox = self.bboxes_id[bbox_id]
            bbox_coord = bbox.coord.to_list()
            if bbox_coord[0] < coord[0]:
                coord[0] = bbox_coord[0]
            if bbox_coord[1] < coord[1]:
                coord[1] = bbox_coord[1]
            if bbox_coord[2] > coord[2]:
                coord[2] = bbox_coord[2]
            if bbox_coord[3] > coord[3]:
                coord[3] = bbox_coord[3]
            self.delete_bbox(bbox)
        
        self.entobox.bboxes = [box for box in self.entobox.bboxes if box not in self.selected]
            
        self.selected = []
        self.new_bbox(*coord)

    def confirm(self, boxid):
        if self.moved:
            self.unselect_all()
            bbox : BBox = self.bboxes_id[boxid]
            bbox.conf = 1
            self.update_bbox_color(bbox)
            self.moved = False
            self.canvas.event_generate("<<OnBBoxModified>>")

    def confirm_selected(self):
        for boxid in self.selected:
            bbox : BBox = self.bboxes_id[boxid]
            bbox.conf = 1
            bbox.is_selected = False #unselect
            self.update_bbox_color(bbox)

        self.selected = []
    
    def reject_selected(self):
        for boxid in self.selected:
            bbox : BBox = self.bboxes_id[boxid]
            bbox.conf = 0
            bbox.is_selected = False #unselect
            self.update_bbox_color(bbox)

        self.selected = []
    
    def group_selected(self, group_label : str):
        if group_label != "":
            for boxid in self.selected:
                bbox : BBox = self.bboxes_id[boxid]
                bbox.group = group_label
    
    def __move_point(self, pointId, boxId, event : tk.Event):
        if self.selecting:
            self.moved = True
            x, y = self.__ratio2image(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            point = self.points_id[pointId]
            
            dx, dy = point.move(x, y)
            self.canvas.move(pointId, dx * self.imscale, dy * self.imscale)

            bbox = self.bboxes_id[boxId]
            self.update_bbox_coords(boxId, bbox.coord)

    def update_bbox_coords(self, itemId : str | int, coords : Coords):
        x1, y1 = self.__ratio2canvas(*coords.first.to_list())
        x2, y2 = self.__ratio2canvas(*coords.second.to_list())
        self.canvas.coords(itemId, x1, y1, x2, y2)        
        
    def update_bbox_color(self, bbox : BBox):
        if bbox.itemId is None:
            self.draw_bbox(bbox)
            return
        if bbox.status() == Status.REJECTED:
            self.delete_bbox(bbox)
            return
        color = bbox.color()

        self.canvas.itemconfig(bbox.itemId, outline=color)
        self.canvas.itemconfig(bbox.coord.first.itemId, outline=color)
        self.canvas.itemconfig(bbox.coord.second.itemId, outline=color)

    def delete_bbox(self, bbox : BBox):
        self.canvas.delete(bbox.itemId, bbox.coord.first.itemId, bbox.coord.second.itemId)
        bbox.itemId = None
        bbox.coord.first.itemId = None
        bbox.coord.second.itemId = None

    def destroy(self):
        for id in self.bboxes_id:
            self.bboxes_id[id].itemId = None
        self.bboxes_id : dict[int | str, BBox] = dict()

        for id in self.points_id:
            self.points_id[id].itemId = None
        self.points_id : dict[int | str, Point] = dict()

        super().destroy()