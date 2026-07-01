import os
import sys
import requests
import inference_pipeline
from shutil import rmtree, move
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import pandas as pd
import json

from src.components.entobox_canvas import EntoboxCanvas
from src.models.boxes import BBox, EntoBox
from src.consts import *


class GUI:

    started = False
    entoboxes = []
    current = 0
    drawn_bboxes = []
    selected = []
    classes = []
    groups = []
    img_id = None

    source_path = None
    img_path = None
    raw_path = None
    label_path = None

    model = DEFAULT_MODEL
    detection_only = True

    n_img = 0
    crop_margin = 1.1

    drawing = False
    drawing_reason = 0
    draw_coord = None
    draw_indic = None

    al_nbr = 3

    def __init__(self):
        root = tk.Tk()
        root.minsize(300,150)
        root.attributes('-zoomed', True)
        root.title("InsectoVision")
        main_frame = ttk.Frame(root, padding=1)
        main_frame.grid(sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.grid(column=0, row=0, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = None

        self.controls_frame = ttk.Frame(main_frame)
        self.controls_frame.grid(column=1, row=0, sticky="ns")

        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0)
        main_frame.grid_rowconfigure(0, weight=1)
        self.root = root

        self.make_menubar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close);
        self.root.focus()
        self.root.mainloop()

    def make_menubar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        filemenu = tk.Menu(menubar,tearoff=False)
        filemenu.add_command(label="Select image folder...",command=self.choose_input)
        filemenu.add_command(label="Create folder from URL list...",command=self.choose_url_list_input)
        filemenu.add_command(label="Open selected images",command=self.load_images)
        filemenu.add_command(label="Scan selected images",command=self.run_inference)
        filemenu.add_separator()
        filemenu.add_command(label="Quick open...",command=self.quick_open)
        filemenu.add_command(label="Quick open from URLs...",command= lambda : self.quick_open(use_url=True))
        filemenu.add_separator()
        #filemenu.add_command(label="Select annotation save folder...",command=self.choose_output)
        filemenu.add_command(label="Summarize saved boxes",command=self.summarize)
        filemenu.add_command(label="Crop specimens from current box",command=self.crop_current)
        filemenu.add_separator()
        filemenu.add_command(label="Parameters",command=self.model_params)
        menubar.add_cascade(label="File",menu=filemenu)

        editmenu = tk.Menu(menubar,tearoff=False)
        editmenu.add_command(label="Edit boxes Ctrl+E",command=self.edit_mode)
        editmenu.add_command(label="Drawing boxes Ctrl+B",command=self.draw_mode)
        
        editmenu.add_command(label="Combine selected bboxes",command=self.combine)
        #editmenu.add_command(label="Add label",command=self.add_label)
        menubar.add_cascade(label="Edit",menu=editmenu)

        #aimenu = Menu(menubar,tearoff=False)
        #aimenu.add_command(label="Open images for active learning",command=self.open_AL)
        #aimenu.add_command(label="Retrain model with new annotations")
        #menubar.add_cascade(label="AI",menu=aimenu)

    def draw_mode(self):
        if self.canvas != None:
            self.canvas.set_to_drawing()
    def edit_mode(self):
        if self.canvas != None:
            self.canvas.set_to_selecting()

    def choose_input(self):
        path = fd.askdirectory(initialdir="~/Elliot")

        if not os.path.exists(os.path.join(path,"images")):
            return

        for folder in ["labels","raw_ai_labels"]:
            if not os.path.exists(os.path.join(path,folder)):
                os.mkdir(os.path.join(path,folder))


        self.img_path = os.path.join(path,"images")
        self.label_path = os.path.join(path,"labels")
        self.raw_path = os.path.join(path,"raw_ai_labels")
        self.source_path = path


        for file in os.listdir(path):
            if file.endswith(".jpg"):
                move(os.path.join(path,file),self.img_path)
            if file.endswith(".txt"):
                move(os.path.join(path,file),self.label_path)
        
        self.root.title("Insectovision - "+self.img_path)

    def choose_url_list_input(self):
        
        txt_file = fd.askopenfile(mode='r',filetypes=[("Text file","*.txt")])
        if txt_file is None:
            return
        
        i = 1
        while os.path.exists(os.path.join(self.label_path,"downloaded_images_"+str(i))):
            i += 1
        path = os.path.join(self.label_path,"downloaded_images_"+str(i))
        os.makedirs(path)
        os.mkdir(os.path.join(path,"images"))
        os.mkdir(os.path.join(path,"labels"))
        os.mkdir(os.path.join(path,"raw_ai_labels"))

        self.img_path = os.path.join(path,"images")
        self.label_path = os.path.join(path,"labels")
        self.raw_path = os.path.join(path,"raw_ai_labels")
        self.source_path = path

        cnt = 1
        for line in txt_file:
            g = requests.get(line.strip("\n"))
            file = open(os.path.join(self.img_path,"image_"+str(cnt)+".jpg"),"wb")
            file.write(g.content)
            file.close()
            cnt += 1
    
    def load_images(self,names = None):
        if(not self.started):
           self.start()

        self.entoboxes = []
        if names is None:
            names = os.listdir(self.img_path)

        need_inf = False

        for entry in names:
            # TODO : select images that are not only .jpg
            if(entry.endswith(".jpg")):
                img_path = os.path.join(self.img_path,entry)
                if(os.path.exists(os.path.join(self.source_path,"labels",entry[:len(entry)-4]+".txt"))):
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],img_path,self.label_path))

                elif(os.path.exists(os.path.join(self.source_path,"raw_ai_labels",entry[:len(entry)-4]+".txt"))):
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],img_path,self.raw_path))

                else:
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],img_path))
                    need_inf = True
        
        self.n_img = len(self.entoboxes)


        
        self.current = 0
        self.set_index(0)

        return need_inf

    def run_inference(self):
        sys.argv = ["inference_pipeline.py", '--input_folder' , self.img_path, "--write_conf","--silent","--model"]
        sys.argv.append(self.model)
        if self.detection_only: 
            sys.argv.append("--detection_only")
        
        print(sys.argv)

        for file in os.listdir(self.raw_path):
            if file.endswith(".txt"):
                os.remove(os.path.join(self.raw_path,file))

        self.root.title("InsectoVision - Scanning...")
        
        args = inference_pipeline.parse_args()
        inference_pipeline.main(args)

        for file in os.listdir("output"):
            move(os.path.join(os.getcwd(),"output",file),os.path.join(self.source_path,"raw_ai_labels"))

        os.rmdir("output")

        self.root.title("InsectoVision - "+self.entoboxes[self.current].name)

        for eb in self.entoboxes:
            eb.get_bboxes(self.raw_path)
        
        if self.entoboxes != []:
            self.show_image()

    def quick_open(self,use_url = False):
        if use_url:
            self.choose_url_list_input()
        else: 
            self.choose_input()
        
        need_inf = self.load_images()
        
        if need_inf: self.run_inference()

    def open_AL(self):
        self.entoboxes = []
        
        chosen = os.listdir(self.img_path)

        self.load_images(chosen[:self.al_nbr])

    def model_params(self):
        
        def select_model():
            self.model = fd.askopenfilename(initialdir="model",filetypes=[("PyTorch model file",".pt")])
            model_label.config(text="Model: "+ self.model)

        param_window = tk.Toplevel()
        param_window.config(width=600,height=100)
        param_window.geometry('+500+500')
        tfrm = ttk.Frame(param_window, padding=5)
        tfrm.grid()

        model_label = ttk.Label(tfrm,text="Model: "+ self.model)
        model_label.grid(row=0,column=0)
        ttk.Button(tfrm,text="Select model",command=select_model).grid(row=0,column=1)
        
        
        detonly = tk.BooleanVar()
        detonly.set(self.detection_only)
        ttk.Checkbutton(tfrm,text="Post-detection classifier",variable=detonly,onvalue=False,offvalue=True).grid(row=1,column=0)
        

        def conf_label():
            self.detection_only = detonly.get()
            param_window.destroy()
        
        def reset_params():
            self.model = DEFAULT_MODEL
            model_label.config(text="Model: "+ self.model)
            self.detection_only = True
            detonly.set(self.detection_only)
        
        ttk.Button(tfrm,text="Reset Default",command=reset_params).grid(row=2,column=0)
        ttk.Button(tfrm,text="Confirm",command=conf_label).grid(row=2,column=1)
        
        param_window.focus()
    
    def summarize(self):
        entobox = self.current_entobox()
        if entobox is None:
            return
        self.summarize_entobox(entobox)

    def summarize_entobox(self,box : EntoBox):
        entobox = self.current_entobox()
        if entobox is None:
            return
        
        self.get_classes()
        
        accepted_bboxes = filter(lambda bbox : bbox.status() == Status.CONFIRMED or bbox.status() == Status.SURE, box.bboxes) # Get only accepted bboxes
        totals = defaultdict(int)
        groups = defaultdict(lambda : defaultdict(int))
        for bbox in accepted_bboxes:
            totals[bbox.label] += 1
            group = bbox.group if bbox.group != "" else "Default"
            groups[group][bbox.label] += 1

        sf = open(os.path.join(self.source_path,"summary.csv"),"w")
        sf.write("Group; Class; Amount\n")
        for label in totals.keys():
           sf.write(f"Total; {label}; {totals[label]}\n")
        for group in groups.keys():
            for label in groups[group].keys():
                sf.write(f"{group}; {label}; {groups[group][label]}\n")
        sf.close()

    def summarize_from_label_files(self):
        
        types = defaultdict(int)
        self.get_classes()

        for boxfile in os.listdir(self.label_path):
            if boxfile.endswith(".txt") and boxfile != "classes.txt" and not boxfile.endswith("_tags.txt"):
                b = open(os.path.join(self.label_path,boxfile),"r")
                for line in b:
                    l = line.split()
                    types[l[0]] +=1

        total = 0
        for amount in types.values():
            total += amount


        sf = open(os.path.join(self.source_path,"summary.csv"),"w")
        sf.write("Specimen type, Amount\n")
        sf.write("Total,"+str(total)+"\n")
        for t in types.keys():
            sf.write((self.classes[int(t)])+","+str(types[t])+"\n")
        sf.close()

    def crop_current(self):
        entobox = self.current_entobox()
        if entobox is None:
            return
        self.save(entobox)
        self.crop_bboxes(entobox)

    def crop_all_images(self):
        for entobox in self.entoboxes:
            self.save(entobox)
            self.crop_bboxes(entobox)

    def crop_bboxes(self,box : EntoBox):
        
        self.get_classes()

        
        
        orig_img = Image.open(os.path.join(self.img_path,box.name+".jpg"))
        labeled_img = Image.open(os.path.join(self.img_path,box.name+".jpg"))
        label_no_box_img = Image.open(os.path.join(self.img_path,box.name+".jpg"))
        draw_bbox = ImageDraw.Draw(labeled_img)
        draw_label = ImageDraw.Draw(label_no_box_img)
        width, height = labeled_img.size
        font_scale = min(width, height) * FONT_SCALE
        font = ImageFont.load_default(font_scale)

        

        dirn = os.path.join(self.source_path,"crops",box.name)

        if os.path.exists(dirn):
            rmtree(dirn)
        os.makedirs(dirn)
        default_boxes = os.path.join(dirn,"Default")
        os.makedirs(default_boxes)
        
        cnt = defaultdict(int)

        accepted_bboxes = filter(lambda bbox : bbox.status() in Status.ACCEPTED, box.bboxes) # Get only accepted bboxes
        sorted_bboxes = self.sort_bboxes_by_columns(accepted_bboxes, image_width=box.width)

        groups = defaultdict(lambda : defaultdict(lambda : defaultdict()))

        names = []
        for bbox in sorted_bboxes:
            cnt[bbox.label] += 1
            bbox_name = bbox.label+"_"+str('{:03}'.format(cnt[bbox.label]))
            names.append(bbox_name)
            left, top, right, bottom = bbox.coord.to_list()
            cropped = orig_img.crop((left,top,right,bottom))

            draw_bbox.rectangle(((left, top), (right, bottom)), outline="green", width=WIDTH_LINE*3)
            
            
            
            group_label = "Default"
            group_dir = default_boxes
            if bbox.group != "":
                group_label = bbox.group
                group_dir = os.path.join(dirn, bbox.group)
                os.makedirs(group_dir, exist_ok=True)
            os.makedirs(os.path.join(group_dir,bbox_name), exist_ok=True)
            cropped.save(os.path.join(group_dir,bbox_name,bbox_name)+".jpg","JPEG")
            cropped.close()

            groups[group_label][bbox_name] = bbox.coord.center()

        # draw label boxes on top of everything
        for i, bbox in enumerate(sorted_bboxes):
            # Get the bounding box of the text itself
            bbox_name = names[i].split('_')[-1]
            left, top, right, bottom = bbox.coord.to_list()
            center_w = (right+left)/2
            center_h = (bottom+top)/2

            left_box, top_box, right_box, lower_box = draw_bbox.textbbox((0, 0), bbox_name, font=font)
            text_w = right_box - left_box
            text_h = lower_box - top_box

            # Position the text just above the bounding box
            text_x = left
            text_y = top - text_h - PAD_BOX *2

            center_x = center_w - text_w/2 - PAD_BOX
            center_y = center_h - text_h/2 - PAD_BOX

            # If the label exceeds the top of the image, put it just inside the box instead
            if text_y < 0:
                text_y = top

            # Draw a filled background rectangle for the label
            #draw_bbox.rectangle([text_x, text_y, text_x + text_w + PAD_BOX*2, text_y + text_h + PAD_BOX*2], fill="black")
            draw_label.rectangle([center_x, center_y, center_x + text_w + PAD_BOX*2, text_h + center_h + PAD_BOX*2], fill="white")

            # Draw the text over the label background
            draw_bbox.text((text_x, text_y), bbox_name, fill="red", font=font)
            draw_label.text((center_x + PAD_BOX, center_y), bbox_name, fill="red", font=font)

        
        group_list = list()
        i = 0
        for group in sorted(groups.keys()):
            for bbox_name in sorted(groups[group].keys()):
                group_list.append({
                    "Group": group,
                    "Name" : bbox_name,
                    "X" : groups[group][bbox_name][0] / box.width,
                    "Y" : groups[group][bbox_name][1] / box.height
                })
        group_df = pd.DataFrame(data=group_list,columns=["Group", "Name", "X", "Y"])
                
        group_df.to_csv(os.path.join(dirn, "summary_crops.csv"), sep=";", index=False)
        labeled_img.save(os.path.join(dirn, "box_image.jpg"),"JPEG")
        label_no_box_img.save(os.path.join(dirn, "label_image.jpg"),"JPEG")
        labeled_img.close()
        orig_img.close()

        self.summarize_entobox(box)

    def start(self):
        self.make_interface()
        self.make_thresh()

    def current_entobox(self):
        if self.current < len(self.entoboxes) and self.current >= 0:
            return self.entoboxes[self.current]
        return None

    def make_interface(self):
        #Title and buttons
        self.title_label = ttk.Label(self.controls_frame, text="Image "+str(self.current+1)+" /"+str(self.n_img))
        self.title_label.grid(column=1, row=0)
        self.number_label = ttk.Label(self.controls_frame, text= str(len(self.current_entobox().bboxes) if self.current_entobox() else 0)+" speciments detected",width=23)
        self.number_label.grid(column=2,row=0)
        ttk.Button(self.controls_frame,text="Previous", command=self.prev,width=BWIDTH).grid(column=1, row=1,padx=PADX)
        ttk.Button(self.controls_frame,text="Next", command=self.next,width=BWIDTH).grid(column=2, row=1,padx=PADX)
        ttk.Button(self.controls_frame,text="Good detection",command=self.confirm_selected,width=BWIDTH).grid(column=1,row=2,padx=PADX)
        ttk.Button(self.controls_frame,text="Bad detection",command=self.reject_selected,width=BWIDTH).grid(column=2,row=2,padx=PADX)
        ttk.Button(self.controls_frame,text="Combine boxes",command=self.combine,width=BWIDTH).grid(column=1,row=3,padx=PADX)
        ttk.Button(self.controls_frame,text="New box",command=self.draw_mode,width=BWIDTH).grid(column=2,row=3,padx=PADX)
        ttk.Button(self.controls_frame,text="Group boxes",command=self.group_boxes,width=BWIDTH).grid(column=1,row=4,padx=PADX)

        ttk.Button(self.controls_frame,text="Save crops",command=self.crop_current,width=BWIDTH).grid(column=1,row=6, padx=PADX)
        ttk.Button(self.controls_frame,text="Save yolo labels",command=self.save_current,width=BWIDTH).grid(column=2,row=6, padx=PADX)
        ttk.Button(self.controls_frame,text="Save all crops",command=self.crop_all_images,width=BWIDTH).grid(column=1,row=7, padx=PADX)
        self.save_label = ttk.Label(self.controls_frame)
        self.save_label.grid(column=1,row=5,padx=PADX)

        self.root.bind("<Delete>", lambda e : self.reject_selected())
        self.root.bind("<Return>", lambda e : self.confirm_selected())

        self.root.bind("<Control-e>", lambda e : self.edit_mode())
        self.root.bind("<Control-b>", lambda e : self.draw_mode())

    def make_thresh(self):
        self.thresh_label = ttk.Label(self.controls_frame, text= f"Confidence threshold: {int(100*DEFAULT_CONF)}%",width=26)
        self.thresh_label.grid(column=1,row=5,padx=PADX)

        self.thresh_scale = ttk.Scale(self.controls_frame, from_=1,to=100,command=self.update_thresh)
        self.thresh_scale.grid(column=2,row=5,padx=PADX)
        self.thresh_scale.set(100*DEFAULT_CONF)
    
    def draw_bbox(self, bbox):
        self.canvas.draw_bbox(bbox)
        
    def redraw_bbox(self,bbox):
        self.canvas.delete_bbox(bbox)
        self.draw_bbox(bbox)

    def next(self):
        self.set_index(self.current+1)

    def prev(self):
        self.set_index(self.current-1)

    def set_index(self, n : int):
        
        if self.n_img == 0:
            self.current = 0
            return
        self.current = (n + self.n_img) % self.n_img #Wraps around when going next/previous
        self.show_image()

    def show_image(self):
        self.close_image()

        entobox : EntoBox = self.current_entobox()

        self.save_label.config(text="")
        self.title_label.config(text="Image "+str(self.current+1)+" /"+str(self.n_img))

        self.selected = []
        self.canvas_frame.update_idletasks()
        self.canvas = EntoboxCanvas(self.canvas_frame, entobox)
        self.canvas.grid(column=0,row=0,sticky="nsew")

        # draw bboxes
        self.canvas.draw_all_bboxes()
        self.canvas.canvas.bind("<<OnBBoxModified>>", self.update_count)

        self.set_thresh()
        self.root.title("InsectoVision - "+entobox.name)

    def close_image(self):
        """ Close image """
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None
    
    def unselect(self):
        for bbox in self.selected:
            self.redraw_bbox(bbox)
        self.selected = []

    def confirm_selected(self):
        if self.canvas:
            self.canvas.confirm_selected()
            self.update_count()
    
    def reject_selected(self):
        if self.canvas:
            self.canvas.reject_selected()
            self.update_count()



    def update_thresh(self,val):
        entobox : EntoBox = self.current_entobox()
        if entobox is None:
            return

        val = int(float(val))
        entobox.conf_threshold = float(val)/100
        self.set_thresh(False)
        
    def set_thresh(self, update_thresh_scale = True):
        entobox : EntoBox = self.current_entobox()
        if entobox is None:
            return
        self.thresh_label.config(text= f"Confidence threshold: {int(entobox.conf_threshold*100)}%")

        for bbox in entobox.bboxes:
            if bbox.status() in Status.NO_UPDATE:
                continue
            self.canvas.update_bbox_color(bbox)
        #self.set_index(self.current) #Redraws current entobox 
        if update_thresh_scale:
            self.thresh_scale.set(int(entobox.conf_threshold*100))
        self.update_count()

    def update_count(self, event=None):
        entobox = self.current_entobox()
        if entobox is None:
            return
        cnt = 0
        for bbox in entobox.bboxes:
            if bbox.status() in Status.ACCEPTED:
                cnt += 1
        self.number_label.config(text= str(cnt)+" speciments detected")
    
    def combine(self):
        self.canvas.combine_select_bboxes()

    def group_boxes(self):
        label_window = tk.Toplevel()
        label_window.configure(width=1000, height=1000)
        label_window.geometry('+1000+1000')
        tfrm = ttk.Frame(label_window, padding=5)
        tfrm.grid()
        tfrm.rowconfigure(2, weight=1)
        ttk.Label(tfrm,text="Enter label name").grid(row=0,column=0)
        e = ttk.Entry(tfrm)
        e.grid(row=1,column=0)
        e.focus()

        tree_frame = ttk.Frame(tfrm)
        tree_frame.grid(row=2,column=0)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)       

        tree = ttk.Treeview(tree_frame, columns=("Items",), show="headings", selectmode="browse")
        # Add items
        tree.heading(0, text="Existing Groups")
        for group in self.groups:
            tree.insert("", tk.END, values=(group,))

        tree.grid(row=0, column=0)

        # Create a Scrollbar
        vert_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        vert_scroll.grid(row=0, column=1)
        # Configure the Treeview to use the scrollbar
        tree.configure(yscrollcommand=vert_scroll.set)
        
        hor_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        hor_scroll.grid(row=1, column=0)
        hor_scroll.unbind("Ctrl")
        # Configure the Treeview to use the scrollbar
        tree.configure(xscrollcommand=hor_scroll.set)
        
        def change_group(a):
            e.delete(0,tk.END)
            item_id = tree.selection()[0]
            e.insert(0, tree.item(item_id)["values"][0])

        def conf_label(a = 0):  #dummy argument, needed to bind to <Return>
            group_label = e.get()
            self.canvas.group_selected(group_label)
            if not group_label in self.groups:
                self.groups.append(group_label)
            label_window.destroy()

        # Bind selection event
        tree.bind("<<TreeviewSelect>>", change_group)
        
        label_window.bind('<Return>',conf_label) #<Return> is the Enter key
        ttk.Button(tfrm,text="Ok",command=conf_label).grid(row=3,column=0)


    def get_classes(self):
        name = os.path.join(self.label_path,"classes.txt")

        if not os.path.isfile(name):
            open(name,"w")
        else:
            lf = open(name,"r")
            self.classes = []
            for line in lf:
                self.classes.append(line.strip("\n"))
            lf.close()

    def sort_bboxes_by_columns(self, bboxes, image_width=None):
        """Group bboxes into columns (left-to-right) and sort each column by vertical position.

        bboxes: iterable of BBox
        image_width: optional image width to compute tolerance for column grouping
        """
        boxes = list(bboxes)
        if not boxes:
            return []

        centers = []
        for bb in boxes:
            cx, cy = bb.coord.center()
            left, top, right, bottom = bb.coord.to_list()
            bw = right - left
            centers.append((bb, cx, cy, bw))

        centers.sort(key=lambda x: x[1])  # sort by center x

        avg_bw = sum(c[3] for c in centers) / len(centers)
        tol = max(avg_bw * 0.8, (image_width or boxes[0].parent.width) / 20)

        columns = []
        for b, cx, cy, bw in centers:
            best_col = None
            best_dist = None
            for col in columns:
                dist = abs(cx - col['mean_x'])
                if dist <= tol and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_col = col
            if best_col is None:
                columns.append({'mean_x': cx, 'boxes': [(b, cx, cy)]})
            else:
                best_col['boxes'].append((b, cx, cy))
                xs = [it[1] for it in best_col['boxes']]
                best_col['mean_x'] = sum(xs) / len(xs)

        columns.sort(key=lambda c: c['mean_x'])

        result = []
        for col in columns:
            col['boxes'].sort(key=lambda it: it[2])  # sort by center y
            result.extend([it[0] for it in col['boxes']])

        return result

    def save_current(self):
        entobox : EntoBox = self.current_entobox()
        if entobox is None:
            self.popup("Save failed: No image loaded")
            return
        self.save(entobox)
            
    def save(self, entobox):
        """
        missing = False
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status() == Status.DOUBT:
                missing = True
                break
        if missing:
            self.save_label.config(text="Save failed: Unvalidated boxes remaining")
            return
        """
        
        #uncomment to force user to confirm all incorrect bboxes

        if not os.path.isdir(self.label_path):
            #self.save_label.config(text="Save failed: Save folder does not exist")
            self.popup("Save failed: Save folder does not exist")
            return

        self.get_classes()
        
        class_file = open(os.path.join(self.label_path,"classes.txt"),"a")
        yolo_file = open(os.path.join(self.label_path,entobox.name+".txt"),"w")
        save_file = open(os.path.join(self.label_path,entobox.name+".json"),"w")
        boxes = list() 

        bboxes_not_rejected = filter(lambda bbox : bbox.status() != Status.REJECTED, entobox.bboxes)
        sorted_bboxes : list[BBox] = self.sort_bboxes_by_columns(bboxes_not_rejected, image_width=entobox.width)
        cnt = defaultdict(int)

        i = 0
        for bbox in sorted_bboxes:
            i +=1
            bbox_name = f"{bbox.label}_{i}"
            if bbox.status() in Status.ACCEPTED:
                cnt[bbox.label] += 1
                bbox_name = bbox.label+"_"+str('{:03}'.format(cnt[bbox.label]))
                if bbox.label not in self.classes:
                    class_file.write(bbox.label+"\n")
                    self.classes.append(bbox.label)
                cnum = self.classes.index(bbox.label)
                # Write only accepted bbox to yolo file
                yolo_file.write(str(cnum)+" "+" ".join(str(x) for x in bbox.to_yolo(entobox.width, entobox.height))+ "\n")
            boxes.append({
                "name" : bbox_name,
                "group" : bbox.group,
                "label" : bbox.label,
                "conf" : bbox.conf,
                "position" : bbox.to_yolo(entobox.width, entobox.height)
            })
        
        save_json = {
            "image" : entobox.image,
            "width" : entobox.width,
            "height" : entobox.height,
            "conf": entobox.conf_threshold,
            "bboxes" : boxes
        }
        
        json.dump(save_json, save_file, indent=3)

        yolo_file.close()
        save_file.close()
        class_file.close()

        #self.save_label.config(text="Save Successful")
        self.popup("Save Successful")
        
    def popup(self,text):
        popup_window = tk.Toplevel()
        popup_window.config(width=600,height=100)
        popup_window.geometry('+500+500')
        tfrm = ttk.Frame(popup_window, padding=5)
        tfrm.grid()
        ttk.Label(tfrm,text=text).grid(row=0,column=0)
        popup_window.focus()

        def conf(a = 0):  #dummy argument, needed to bind to <Return>
            popup_window.destroy()
        
        popup_window.bind('<Return>',conf) #<Return> is the Enter key
        ttk.Button(tfrm,text="Ok",command=conf).grid(row=1,column=0)

    def on_close(self):
        self.root.destroy()
        if os.path.exists("output"):
            rmtree("output")


if __name__ == "__main__":

    gui = GUI()
