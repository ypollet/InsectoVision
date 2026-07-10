import tkinter as tk
from tkinter import ttk

class GroupTopup(tk.Toplevel):
    def __init__(self, groups, callback):
        super().__init__()
        self.groups = groups
        self.callback = callback

        self.configure(width=1000, height=1000)
        self.geometry('+1000+1000')
        tfrm = ttk.Frame(self, padding=5)
        tfrm.grid(sticky="nsew")
        tfrm.rowconfigure(2, weight=1)
        tfrm.columnconfigure(0, weight=1)
        ttk.Label(tfrm,text="Enter label name").grid(row=0,column=0)
        self.e = ttk.Entry(tfrm)
        self.e.grid(row=1,column=0, sticky="ew")
        self.e.focus()

        tree_frame = ttk.Frame(tfrm)
        tree_frame.grid(row=2,column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, columns=("Items",), show="headings", selectmode="browse")
        # Add items
        self.tree.heading(0, text="Existing Groups")
        for group in self.groups:
            self.tree.insert("", tk.END, values=(group,))

        self.tree.grid(row=0, column=0, sticky="nsew")

        # Create a Scrollbar
        vert_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        vert_scroll.grid(row=0, column=1, sticky="ns")
        # Configure the Treeview to use the scrollbar
        self.tree.configure(yscrollcommand=vert_scroll.set)
        
        hor_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        hor_scroll.grid(row=1, column=0, sticky="ew")
        hor_scroll.unbind("Ctrl")
        # Configure the Treeview to use the scrollbar
        self.tree.configure(xscrollcommand=hor_scroll.set)
        
        

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.change_group)
        
        self.bind('<Return>', self.on_ok) #<Return> is the Enter key
        ttk.Button(tfrm,text="Ok",command=self.on_ok).grid(row=3,column=0, sticky="")
    
    def change_group(self, a):
            self.e.delete(0,tk.END)
            item_id = self.tree.selection()[0]
            self.e.insert(0, self.tree.item(item_id)["values"][0])


    def on_ok(self, a=0):
        self.callback(self.e.get()) # send the back to the other class.
        self.destroy()