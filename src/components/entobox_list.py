import tkinter as tk
from tkinter import ttk

import math

from src.models.boxes import EntoBox
from src.consts import MAX_SIZE_ENTOBOX_LIST

from src.components.scrollbar import AutoScrollbar

class EntoboxItem(ttk.Frame):
    def __init__(self, master, entobox : EntoBox, **kwargs):
        super().__init__(master, **kwargs)
        self.var = entobox.saved

        self.filename = entobox.name

        self.columnconfigure(0, weight=1)

        self.check = ttk.Checkbutton(self, text=self.filename,variable=self.var, state="disabled")
        self.check.grid(row=0, column=0, sticky="ew")
    
    def bind(self, command, callback):
        self.check.bind(command, callback)

class EntoboxList(ttk.Frame):
    def __init__(self, master, entoboxes : list[EntoBox] = [], height=8, **kwargs):
        super().__init__(master, **kwargs)
        self.rows = []


        self.__imframe = ttk.Frame(self)  # placeholder of the ImageFrame object
        self.__imframe.pack(expand=True, fill='both')
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.rowconfigure(1, weight=0)
        self.__imframe.columnconfigure(0, weight=1)
        self.__imframe.columnconfigure(1, weight=0)
        self.__imframe.update_idletasks()

        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nswe')

        self.scrollbar = AutoScrollbar(self.__imframe, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        
        

        s = ttk.Style()
        s.configure('black.TFrame', background='black')

        self.inner_frame = ttk.Frame(self.canvas, style="black.TFrame")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame.bind('<Enter>', self._bound_to_mousewheel)
        self.inner_frame.bind('<Leave>', self._unbound_to_mousewheel)

        

        self.add_items(entoboxes)

        self.configure_canvas_size()

    def _bound_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind_all("<Button-4>", self._on_mouse_wheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mouse_wheel_linux)

    def _unbound_to_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def configure_canvas_size(self):
        self.inner_frame.update_idletasks()
        print(f"Configure : {min(self.inner_frame.winfo_height(), MAX_SIZE_ENTOBOX_LIST)}")
        self.canvas.configure(height=min(self.inner_frame.winfo_height(), MAX_SIZE_ENTOBOX_LIST))
        if self.inner_frame.winfo_height() < MAX_SIZE_ENTOBOX_LIST:
            self.scrollbar.grid_remove()
    
    def __len__(self):
        return len(self.rows)
        

    def reset(self):
        self.inner_frame.update_idletasks()
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.rows = []
    
    def add_items(self, entoboxes : list[EntoBox]):
        for entobox in entoboxes:
            print(f"Adding {entobox.name}")
            row = EntoboxItem(self.inner_frame, entobox)
            index = len(self.rows)
            row.bind("<Double-Button-1>", lambda e, index=index: self.set_index(index))
            row.pack(fill="x")
            self.rows.append(row)
        self.configure_canvas_size()
        print(f"After configure")
    
    def add_item(self, entobox_item : EntoboxItem):

        entobox_item.pack(fill="x")
        self.rows.append(entobox_item)
        self.configure_canvas_size()
    
    def set_index(self, index):
        self.event_generate("<<Set-Index>>", x=index)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.height if event.height > MAX_SIZE_ENTOBOX_LIST else event.height)
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mouse_wheel_linux(self, event):
        if self.inner_frame.winfo_height() < self.canvas.winfo_height():
            return
        delta = 1 if event.num == 5 else -1
        self.canvas.yview_scroll(delta*5, "units")
    
    def _on_mouse_wheel(self, event):
        if self.inner_frame.winfo_height() < self.canvas.winfo_height():
            return
        self.canvas.yview_scroll(-math.copysign(5, event.delta), "units")
    
    def set_checked(self, index, value):
        if 0 <= index < len(self.rows):
            self.rows[index].set_checked(bool(value))

    def is_checked(self, index):
        if 0 <= index < len(self.rows):
            return self.rows[index][1].get()
        return False