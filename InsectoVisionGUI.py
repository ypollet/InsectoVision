import os
import sys
import requests
import inference_pipeline
from shutil import rmtree, move
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import ImageTk, Image

from src.components.canvas import CanvasImage
from src.models.boxes import BBox, EntoBox
from src.consts import *


class GUI:

    started = False
    entoboxes = []
    current = 0
    drawn_bboxes = []
    selected = []
    classes = []
    img_id = None

    source_path = None
    img_path = None
    raw_path = None
    label_path = None

    model = DEFAULT_MODEL
    detection_only = True

    n_img = 0
    conf_threshold = 0.85
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
        editmenu.add_command(label="New specimen bbox",command=self.start_draw)
        editmenu.add_command(label="New tag bbox",command=self.start_draw_tag)
        editmenu.add_command(label="Combine selected bboxes",command=self.combine)
        editmenu.add_command(label="Add label",command=self.add_label)
        menubar.add_cascade(label="Edit",menu=editmenu)

        #aimenu = Menu(menubar,tearoff=False)
        #aimenu.add_command(label="Open images for active learning",command=self.open_AL)
        #aimenu.add_command(label="Retrain model with new annotations")
        #menubar.add_cascade(label="AI",menu=aimenu)
        
    def choose_input(self):
        path = fd.askdirectory()

        for folder in ["images","labels","raw_ai_labels"]:
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
        #print(self.img_path)

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

        print(f"{len(self.entoboxes)} images")

        
        self.current = 0
        self.set_index(0)

        return need_inf

    def run_inference(self):
        #print("img_path = "+self.img_path)
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
            #print(file)
            move(os.path.join(os.getcwd(),"output",file),os.path.join(self.source_path,"raw_ai_labels"))

        os.rmdir("output")

        self.root.title("InsectoVision - "+self.entoboxes[self.current].name)
        #print("inference done")

        for eb in self.entoboxes:
            eb.get_bboxes(self.raw_path)
        
        if self.entoboxes != []:
            self.show_image(self.current)

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
        
        types = {}
        self.get_classes()

        for boxfile in os.listdir(self.label_path):
            if boxfile.endswith(".txt") and boxfile != "classes.txt" and not boxfile.endswith("_tags.txt"):
                b = open(os.path.join(self.label_path,boxfile),"r")
                for line in b:
                    l = line.split()
                    if l[0] not in types.keys():
                        types[l[0]] = 1
                    else:
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
        self.crop_bboxes(entobox)

    def crop_bboxes(self,box):
        
        self.get_classes()

        yolo_name = os.path.join(self.label_path,box.name+".txt")
        yolot_name = os.path.join(self.label_path,box.name+"_tags.txt")
        if not os.path.isfile(yolo_name):
            self.popup("Crop failed: Yolo file missing")
            return
        
        yolof = open(yolo_name,"r")
        img = Image.open(os.path.join(self.img_path,box.name+".jpg"))
        

        dirn = os.path.join(self.source_path,"crops",box.name) + "_crops"
        if os.path.exists(dirn):
            rmtree(dirn)
        os.makedirs(dirn)

        cnt = {}
        for line in yolof:
            la = line.split(" ")[0:6]                                         
            [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
            [w,h] = [w*self.crop_margin,h*self.crop_margin]
                        
            top = max(int((y-h/2)*img.size[1]),0)
            bottom = min(int((y+h/2)*img.size[1]),img.size[1])
            left = max(int((x-w/2)*img.size[0]),0)
            right = min(int((x+w/2)*img.size[0]),img.size[0])
            
            cropped = img.crop((left,top,right,bottom))
            if la[0] not in cnt.keys():
                cnt[la[0]] = 1
            else:
                cnt[la[0]] += 1
            imgn = self.classes[int(la[0])]+"_"+str(cnt[la[0]])
            cropped.save(os.path.join(dirn,imgn)+".jpg","JPEG")
        yolof.close()
        
        if os.path.isfile(yolot_name):
            yolotf = open(yolot_name,"r")
            cnt2 = {}
            for line in yolotf:
                la = line.split(" ")[0:6]   
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                [w,h] = [w*self.crop_margin,h*self.crop_margin]
                            
                top = max(int((y-h/2)*img.size[1]),0)
                bottom = min(int((y+h/2)*img.size[1]),img.size[1])
                left = max(int((x-w/2)*img.size[0]),0)
                right = min(int((x+w/2)*img.size[0]),img.size[0])
                
                cropped = img.crop((left,top,right,bottom))
                if la[0] not in cnt2.keys():
                    cnt2[la[0]] = 1
                else:
                    cnt2[la[0]] += 1
                imgn = self.classes[int(la[0])]+"_tag_"+str(cnt2[la[0]])
                cropped.save(os.path.join(dirn,imgn)+".jpg","JPEG")
            yolotf.close()

        img.close()

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
        ttk.Button(self.controls_frame,text="Good detection",command=self.rate_g,width=BWIDTH).grid(column=1,row=2,padx=PADX)
        ttk.Button(self.controls_frame,text="Bad detection",command=self.rate_b,width=BWIDTH).grid(column=2,row=2,padx=PADX)
        ttk.Button(self.controls_frame,text="Combine boxes",command=self.combine,width=BWIDTH).grid(column=1,row=3,padx=PADX)
        ttk.Button(self.controls_frame,text="New box",command=self.start_draw,width=BWIDTH).grid(column=2,row=3,padx=PADX)
        ttk.Button(self.controls_frame,text="Add label",command=self.add_label,width=BWIDTH).grid(column=1,row=4,padx=PADX)

        ttk.Button(self.controls_frame,text="Save",command=self.save,width=BWIDTH).grid(column=2,row=6,columnspan=2, padx=PADX)
        self.save_label = ttk.Label(self.controls_frame)
        self.save_label.grid(column=1,row=5,padx=PADX)

    def make_thresh(self):
        self.thresh_label = ttk.Label(self.controls_frame,width=26)
        self.thresh_label.grid(column=1,row=5,padx=PADX)

        self.thresh_scale = ttk.Scale(self.controls_frame, from_=0,to=100,command=self.update_thresh)
        self.thresh_scale.grid(column=2,row=5,padx=PADX)
        self.thresh_scale.set(100*self.conf_threshold)
    
    def draw_bbox(self, bbox):
        bbox.update_status(self.conf_threshold)
        boxid = self.canvas.canvas.create_rectangle(bbox.coord.x1,bbox.coord.y1,bbox.coord.x2,bbox.coord.y2,outline=COLORS[bbox.status],width=2,tags=["bbox"])
        bbox.itemId = boxid
        
    def redraw_bbox(self,bbox):
        self.canvas.delete(bbox.itemId)
        self.draw_bbox(bbox)

    def next(self):
        self.set_index(self.current+1)

    def prev(self):
        self.set_index(self.current-1)

    def set_index(self, n : int):
        if self.n_img == 0:
            return
        self.current = (n + self.n_img) % self.n_img #Wraps around when going next/previous
        self.show_image()

    def show_image(self):
        self.save_label.config(text="")
        self.title_label.config(text="Image "+str(self.current+1)+" /"+str(self.n_img))

        self.selected = []
        self.canvas = CanvasImage(self.canvas_frame, self.current_entobox().image)
        self.canvas.grid(column=0,row=0,sticky="nsew")


        for bbox in self.current_entobox().bboxes:
            self.canvas.draw_bbox(bbox)

        self.update_count()
        self.root.title("InsectoVision - "+self.current_entobox().name)


    def unselect(self):
        for bbox in self.selected:
            self.redraw_bbox(bbox)
        self.selected = []

    def rate_g(self):
        for box in self.selected:
            box.status = CONFIRMED
            self.redraw_bbox(box)
        self.unselect()
        self.update_count()
    def rate_b(self):
        for box in self.selected:
            box.status = REJECTED
            self.redraw_bbox(box)
        self.unselect()
        self.update_count()


    def update_thresh(self,val):
        entobox = self.current_entobox()
        if entobox is None:
            return

        val = int(float(val))
        self.conf_threshold = float(val)/100
        self.thresh_label.config(text= "Confidence threshold: "+ str(val)+"%")
        for bbox in entobox.bboxes:
            if bbox.status == DOUBT and bbox.conf > self.conf_threshold:
                bbox.status = SURE
            elif bbox.status == SURE and bbox.conf < self.conf_threshold:
                bbox.status = DOUBT 
        self.set_index(self.current) #Redraws current entobox 
        self.update_count()

    def update_count(self):
        entobox = self.current_entobox()
        if entobox is None:
            return
        cnt = 0
        for bbox in entobox.bboxes:
            if bbox.status == CONFIRMED or bbox.status == SURE:
                cnt += 1
        self.number_label.config(text= str(cnt)+" speciments detected")


    
        

    

    def combine(self):
        if len(self.selected)<2:
            return
        coord = self.selected[0].coord.copy()
        for bbox in self.selected:
            if bbox.coord[0] < coord[0]:
                coord[0] = bbox.coord[0]
            if bbox.coord[1] < coord[1]:
                coord[1] = bbox.coord[1]
            if bbox.coord[2] > coord[2]:
                coord[2] = bbox.coord[2]
            if bbox.coord[3] > coord[3]:
                coord[3] = bbox.coord[3]
        
        entobox = self.current_entobox()
        if entobox is None:
            return
        entobox.bboxes = [box for box in entobox.bboxes if box not in self.selected]
        for bbox in self.selected:
            self.canvas.delete(bbox.itemId)
        self.selected = []
        new = BBox(coord,1,entobox)
        new.status = CONFIRMED
        self.draw_bbox(new)
        entobox.bboxes.append(new)    
        self.unselect()
        self.update_count()


    def add_label(self):
        label_window = tk.Toplevel()
        label_window.config(width=600,height=100)
        label_window.geometry('+500+500')
        tfrm = ttk.Frame(label_window, padding=5)
        tfrm.grid()
        ttk.Label(tfrm,text="Enter label name").grid(row=0,column=0)
        e = ttk.Entry(tfrm)
        e.grid(row=1,column=0)
        e.focus()

        def conf_label(a = 0):  #dummy argument, needed to bind to <Return>
            for bbox in self.selected:
                bbox.label = e.get()
            label_window.destroy()
        
        label_window.bind('<Return>',conf_label) #<Return> is the Enter key
        ttk.Button(tfrm,text="Ok",command=conf_label).grid(row=2,column=0)


    def start_draw(self,reason = NEW_BBOX):
        self.drawing_reason = reason
    def start_draw_tag(self):
        self.start_draw(NEW_TAG)


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
            
    def save(self):
        """
        missing = False
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == DOUBT:
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
        
        lf = open(os.path.join(self.label_path,"classes.txt"),"a")
        f = open(os.path.join(self.label_path,self.entoboxes[self.current].name)+".txt","w")
        tf = None
        

        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status in [SURE,CONFIRMED]:
                if bbox.label not in self.classes:
                    lf.write(bbox.label+"\n")
                    self.classes.append(bbox.label)
                cnum = self.classes.index(bbox.label)
                f.write(str(cnum)+" "+" ".join(str(x) for x in bbox.to_yolo())+ "\n")

            elif bbox.status == TAG: #Write tags to a separate yolo file with the same class system (can be used for tag detection training)
                if tf is None:
                    tf = open(os.path.join(self.label_path,self.entoboxes[self.current].name)+"_tags.txt","w")
                if bbox.label not in self.classes:
                    lf.write(bbox.label+"\n")
                    self.classes.append(bbox.label)
                cnum = self.classes.index(bbox.label)
                tf.write(str(cnum)+" "+" ".join(str(x) for x in bbox.to_yolo())+ "\n") 

        f.close()
        if tf != None: tf.close()
        lf.close()

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

    def get_dim(self,dim):
        x = dim[0]
        y = dim[1]
        self.root.update_idletasks()
        scale = min(self.canvas.winfo_width()/x,self.canvas.winfo_height()/y)
        return(int(x*scale),int(y*scale))

    def on_close(self):
        self.root.destroy()
        if os.path.exists("output"):
            rmtree("output")


if __name__ == "__main__":

    gui = GUI()
