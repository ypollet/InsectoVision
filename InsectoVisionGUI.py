import os
import sys
import requests
import inference_pipeline
from shutil import rmtree, move
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import ImageTk, Image


#Constants
DEFAULT_LABEL = "Insect"
DEFAULT_MODEL = os.path.join("model","final_23.pt")

#Drawing reasons
NEW_BBOX = 1
NEW_TAG = 2
SELECTING = 3

#Bbox status
SURE=1          #AI's confidence is above threshold
DOUBT=2         #AI's confidence is below threshold
CONFIRMED=3     #User has confirmed the bbox is correct
REJECTED=4      #User has confirmed the bbox is incorrect
SELECTED=5      #Currently selected
TAG=6           #User defined paper tag (saved in a separate yolo file, ignored in summary or count)
COLORS =    {SURE:"chartreuse4"
            ,DOUBT:"gold"
            ,CONFIRMED:"green2"
            ,REJECTED:"red"
            ,SELECTED:"blue"
            ,TAG:"purple"
            }


BWIDTH = 15 #Button width
PADX = 5 #x axis padding between buttons/labels
NONCANVASHEIGHT = 150
NONCANVASWIDTH = 30


#One image of an entomology box
#Instance created only when images are loaded into the viewer 
#(not when scanning with neural net) 
class EntoBox:

    def __init__(self,name,img_path,gui,bboxes_path = None):
        self.name = name
        #print("Loading "+name)

        #Get the image
        img = Image.open(os.path.join(img_path,name+".jpg"))        
        self.dim = gui.get_dim((img.size))
        self.image = ImageTk.PhotoImage(img.resize(self.dim))
        img.close()

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
                x1 = int((x-w/2)*self.dim[0])
                x2 = int((x+w/2)*self.dim[0])
                y1 = int((y-h/2)*self.dim[1])
                y2 = int((y+h/2)*self.dim[1])
                if(len(la) == 6):
                    c = float(la[5])
                else:
                    c = 1
                self.bboxes.append(BBox([x1,y1,x2,y2],c,self))
            txt.close()
    
    def show(self,gui):
        gui.selected = []
        gui.img_id = gui.canvas.create_image(int(self.dim[0]/2),int(self.dim[1]/2),image=gui.entoboxes[gui.current].image,tags=["picture"])

        for bbox in self.bboxes:
            bbox.draw(gui)


class BBox:

    status = None
    itemId = None
    label = DEFAULT_LABEL

    def __init__(self,coord,conf,parent):
        self.parent = parent
        self.coord = coord
        self.conf = conf
    
    def draw(self,gui):      
        self.update_status(gui.conf_threshold)
        boxid = gui.canvas.create_rectangle(self.coord[0],self.coord[1],self.coord[2],self.coord[3],outline=COLORS[self.status],width=2,tags=["bbox"])
        self.itemId = boxid
        gui.drawn_bboxes.append(self)
    
    def redraw(self,gui):
        gui.canvas.delete(self.itemId)
        self.draw(gui)

    def to_yolo(self):
        [d0,d1] = self.parent.dim
        [x1,y1,x2,y2] = self.coord
        x = ((float(x2+x1))/2)/d0
        y = ((float(y2+y1))/2)/d1
        w = float(abs(x2-x1))/d0
        h = float(abs(y2-y1))/d1

        return [x,y,w,h]

    def update_status(self,ct):
        if self.status in [None,DOUBT,SURE]:
            if(self.conf < ct):
                self.status = DOUBT
            else:
                self.status = SURE

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
        root = Tk()
        root.minsize(300,150)
        root.title("InsectoVision")
        frm = ttk.Frame(root, padding=1)
        frm.grid()
        self.y_max = root.winfo_screenheight()-NONCANVASHEIGHT
        self.x_max = root.winfo_screenwidth()-NONCANVASWIDTH
        self.root = root
        self.frame = frm

        self.make_menubar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close);
        self.root.focus()
        self.root.mainloop()

    def make_menubar(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        filemenu = Menu(menubar,tearoff=False)
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

        editmenu = Menu(menubar,tearoff=False)
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
        self.entoboxes = []
        if names is None:
            names = os.listdir(self.img_path)

        need_inf = False

        for entry in names:
            if(entry.endswith(".jpg")):
                
                if(os.path.exists(os.path.join(self.source_path,"labels",entry[:len(entry)-4]+".txt"))):
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],self.img_path,self,self.label_path))

                elif(os.path.exists(os.path.join(self.source_path,"raw_ai_labels",entry[:len(entry)-4]+".txt"))):
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],self.img_path,self,self.raw_path))

                else:
                    self.entoboxes.append(EntoBox(entry[:len(entry)-4],self.img_path,self))
                    need_inf = True
        
        self.n_img = len(self.entoboxes)

        if(not self.started):
           self.start()
        
        self.current = 0
        self.show_image(0)

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

        param_window = Toplevel()
        param_window.config(width=600,height=100)
        param_window.geometry('+500+500')
        tfrm = ttk.Frame(param_window, padding=5)
        tfrm.grid()

        model_label = ttk.Label(tfrm,text="Model: "+ self.model)
        model_label.grid(row=0,column=0)
        ttk.Button(tfrm,text="Select model",command=select_model).grid(row=0,column=1)
        
        
        detonly = BooleanVar()
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
        self.crop_bboxes(self.entoboxes[self.current])

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
        self.make_canvas()
        self.make_thresh()

    def make_interface(self):
        #Title and buttons
        self.title_label = ttk.Label(self.frame, text="Image "+str(self.current+1)+" /"+str(self.n_img))
        self.title_label.grid(column=1, row=0)
        self.number_label = ttk.Label(self.frame, text= str(len(self.entoboxes[self.current].bboxes))+" speciments detected",width=23)
        self.number_label.grid(column=2,row=0)
        ttk.Button(self.frame,text="Next", command=self.next,width=BWIDTH).grid(column=2, row=1,padx=PADX)
        ttk.Button(self.frame,text="Previous", command=self.prev,width=BWIDTH).grid(column=1, row=1,padx=PADX)
        ttk.Button(self.frame,text="Good detection",command=self.rate_g,width=BWIDTH).grid(column=5,row=0,padx=PADX)
        ttk.Button(self.frame,text="Bad detection",command=self.rate_b,width=BWIDTH).grid(column=5,row=1,padx=PADX)
        ttk.Button(self.frame,text="Combine boxes",command=self.combine,width=BWIDTH).grid(column=6,row=1,padx=PADX)
        ttk.Button(self.frame,text="New box",command=self.start_draw,width=BWIDTH).grid(column=6,row=0,padx=PADX)
        ttk.Button(self.frame,text="Add label",command=self.add_label,width=BWIDTH).grid(column=7,row=0,padx=PADX)

        ttk.Button(self.frame,text="Save",command=self.save,width=BWIDTH).grid(column=9,row=1,padx=PADX)
        self.save_label = ttk.Label(self.frame)
        self.save_label.grid(column=9,row=0,padx=PADX)

    def make_canvas(self):
        #Canvas for bounding boxes
        self.canvas = Canvas(self.root,height=self.y_max,width=self.x_max)
        self.canvas.grid(column=0,row=2,padx=20)
        self.canvas.bind('<Button-1>',self.on_click)
        self.canvas.bind('<Shift-Button-1>',self.select_many)
        self.canvas.bind('<B1-Motion>',self.on_move_M1_held)
        self.canvas.bind('<ButtonRelease-1>',self.on_M1_release)

        if self.entoboxes != []:
            self.entoboxes[0].show(self)

    def make_thresh(self):
        self.thresh_label = ttk.Label(self.frame,width=26)
        self.thresh_label.grid(column=8,row=0,padx=PADX)

        self.thresh_scale = ttk.Scale(self.frame, from_=0,to=100,command=self.update_thresh)
        self.thresh_scale.grid(column=8,row=1,padx=PADX)
        self.thresh_scale.set(100*self.conf_threshold)



    def next(self):
        self.show_image(self.current+1)

    def prev(self):
        self.show_image(self.current-1)

    def show_image(self,n):
        self.save_label.config(text="")
        self.current = (n)%self.n_img
        self.title_label.config(text="Image "+str(self.current+1)+" /"+str(self.n_img))
        self.canvas.delete(self.img_id)
        for bbox in self.drawn_bboxes:
            self.canvas.delete(bbox.itemId)
        self.entoboxes[self.current].show(self)
        self.update_count()
        self.root.title("InsectoVision - "+self.entoboxes[self.current].name)


    def unselect(self):
        for bbox in self.selected:
            bbox.redraw(self)
        self.selected = []

    def rate_g(self):
        for box in self.selected:
            box.status = CONFIRMED
            box.redraw(self)
        self.unselect()
        self.update_count()
    def rate_b(self):
        for box in self.selected:
            box.status = REJECTED
            box.redraw(self)
        self.unselect()
        self.update_count()


    def update_thresh(self,val):
        val = int(float(val))
        self.conf_threshold = float(val)/100
        self.thresh_label.config(text= "Confidence threshold: "+ str(val)+"%")
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == DOUBT and bbox.conf > self.conf_threshold:
                bbox.status = SURE
            elif bbox.status == SURE and bbox.conf < self.conf_threshold:
                bbox.status = DOUBT 
        self.show_image(self.current) #Redraws current entobox 
        self.update_count()

    def update_count(self):
        cnt = 0
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == CONFIRMED or bbox.status == SURE:
                cnt += 1
        self.number_label.config(text= str(cnt)+" speciments detected")


    def on_click(self,e): 
        if self.drawing == False:
            self.select(e)

    def on_move_M1_held(self,e):
        if not self.drawing:
            self.draw_coord = [e.x,e.y]
            self.drawing = True
        self.canvas.delete(self.draw_indic)
        self.draw_indic = self.canvas.create_rectangle(self.draw_coord[0],self.draw_coord[1],e.x,e.y,outline=COLORS[SELECTED],width=4)
        
    def on_M1_release(self,e):
        if self.drawing:
            self.canvas.delete(self.draw_indic)
            x1 = min(self.draw_coord[0],e.x)
            x2 = max(self.draw_coord[0],e.x)
            y1 = min(self.draw_coord[1],e.y)
            y2 = max(self.draw_coord[1],e.y)

            if self.drawing_reason == SELECTING:
                self.unselect()
                for bbox in self.entoboxes[self.current].bboxes:
                    c = bbox.coord
                    if(x1<c[0] and y1<c[1] and x2>c[2] and y2>c[3]):
                        self.selected.append(bbox)
                        self.canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])

            elif self.drawing_reason in [NEW_BBOX,NEW_TAG]:
                new = BBox([x1,y1,x2,y2],1,self.entoboxes[self.current])

                if self.drawing_reason == NEW_BBOX:
                    new.status = CONFIRMED
                elif self.drawing_reason == NEW_TAG:
                    new.status = TAG
                
                self.entoboxes[self.current].bboxes.append(new)
                new.draw(self)
                self.update_count()


            self.drawing = False
            self.drawing_reason = SELECTING
        

    def select(self,e,cumul = False):
        found = None
        for bbox in self.entoboxes[self.current].bboxes:
            c = bbox.coord 
            if (e.x>c[0] and e.y>c[1] and e.x<c[2] and e.y < c[3] and bbox not in self.selected):
                self.canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])
                found = bbox
                break

        if found != None:
            if cumul:
                self.selected.append(found)
            else:
                self.unselect()
                self.selected = [found]

    def select_many(self,e):
        self.select(e,True)


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
        
        self.entoboxes[self.current].bboxes = [box for box in self.entoboxes[self.current].bboxes if box not in self.selected]
        for bbox in self.selected:
            self.canvas.delete(bbox.itemId)
        self.selected = []
        new = BBox(coord,1,self.entoboxes[self.current])
        new.status = CONFIRMED
        new.draw(self)
        self.entoboxes[self.current].bboxes.append(new)    
        self.unselect()
        self.update_count()


    def add_label(self):
        label_window = Toplevel()
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
        popup_window = Toplevel()
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
        scale = min(self.x_max/x,self.y_max/y)
        return(int(x*scale),int(y*scale))

    def on_close(self):
        self.root.destroy()
        if os.path.exists("output"):
            rmtree("output")


if __name__ == "__main__":

    gui = GUI()
