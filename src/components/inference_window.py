import sys
import os
import multiprocessing
import torch.multiprocessing as mp

import tkinter as tk
from tkinter import ttk

from src.consts import *
from src.models.boxes import EntoBox
from src.models.config import Config

import inference_pipeline

def _run_inference_worker(entoboxes, source_path, model, with_classification, overlap, iou, cancel_event, progress_queue):
    for index, entobox in enumerate(entoboxes):
        if cancel_event.is_set():
            break
        index = entobox[0]
        name = entobox[1]
        image_path = entobox[2]
        progress_queue.put({"type": "progress", "value": index + 1, "name": name})

        label_path = run_single_inference(image_path, source_path, model, with_classification, overlap, iou)
        progress_queue.put({"type": "done", "name": name, "index": index, "label_path": label_path})

    progress_queue.put({"type": "finished"})


def run_single_inference(image_path, source_path, model, with_classification, overlap, iou):
    output_dir = os.path.join(source_path, "raw_ai_labels")
    os.makedirs(output_dir, exist_ok=True)

    sys.argv = [
        "inference_pipeline.py",
        "--input",
        image_path,
        "--output",
        output_dir,
        "--max_overlap",
        str(overlap),
        "--max_iou",
        str(iou),
        "--tiling",
        "--write_conf",
        "--img_size",
        str(DEFAULT_IMG_SIZE),
        "--model",
        model,
    ]
    if not with_classification:
        sys.argv.append("--detection_only")

    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, base_name + ".txt")

class ScanWindow(tk.Toplevel):
    def __init__(self, parent, entoboxes : list[EntoBox], source_path, config : Config):
        super().__init__(parent)
        self.title("Scanning progress")
        self.transient(parent)
        self.attributes("-topmost", True)

        self.model = config.model
        self.with_classification = config.classification
        self.overlap = float(config.max_overlap)
        self.iou = float(config.max_iou)

        self.entoboxes = entoboxes

        self.source_path = source_path

        self.root = ttk.Frame(self, padding=12)
        self.root.pack(fill="both", expand=True)

        label_container = ttk.Frame(self.root)
        label_container.pack(fill="x", expand=True)

        self.scan_progress_label = ttk.Label(label_container, text="Preparing scan...")
        self.scan_progress_label.pack(side=tk.LEFT, anchor="w", pady=(0, 8))

        self.index_label = ttk.Label(label_container, text="")
        self.index_label.pack(side=tk.RIGHT,  anchor="e", pady=(0, 8))

        self.scan_progress_bar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=max(1, len(self.entoboxes)))
        self.scan_progress_bar.pack(fill="x", pady=(0, 8))
        self.scan_progress_bar['value'] = 0

        cancel_button = ttk.Button(self.root, text="Cancel scan", command=self.cancel_scan)
        cancel_button.pack(anchor="e")

        
        self.cancel_event = multiprocessing.Event()
        self.progress_queue = mp.Queue()

        entobox_specs = [(i, entobox.name, entobox.image) for i, entobox in enumerate(self.entoboxes)]
        self.inference_process = mp.Process(
            target=_run_inference_worker,
            args=(entobox_specs, self.source_path, self.model, self.with_classification, self.overlap, self.iou, self.cancel_event, self.progress_queue),
            daemon=True,
        )
        self.inference_process.start()
        self.handle_scan_notifications()
    

    def handle_scan_notifications(self):
        try:
            while True:
                message = self.progress_queue.get_nowait()
                if message.get("type") == "progress":
                    self.update_scan_progress(message["value"], message["name"])
                elif message.get("type") == "done":
                    self.update_entobox_from_output(message["index"], message["label_path"])
                elif message.get("type") == "finished":
                    self.close_scan_progress()
                    self.progress_queue_handler = None
                    return
        except Exception:
            pass
        if self.inference_process is None or not self.inference_process.is_alive():
            self.close_scan_progress()
            self.progress_queue_handler = None
            return

        self.progress_queue_handler = self.root.after(10, self.handle_scan_notifications)

    def update_scan_progress(self, value, name):
        self.scan_progress_bar['value'] = value
        self.scan_progress_label.config(text=f"Processing image {name}")
        self.index_label.config(text=f"{value}/{len(self.entoboxes)}")

    def close_scan_progress(self):
        self.inference_process.terminate()
        self.destroy()

    def cancel_scan(self):
        self.cancel_event.set()
        self.scan_progress_label.config(text="Cancelling scan...")
        self.scan_progress_bar['value'] = 0
        self.close_scan_progress()

    def update_entobox_from_output(self, index, label_path):
        entobox = self.entoboxes[index]
        entobox.ai_labels = label_path
        if not entobox.is_saved():
            entobox.load_from_file(label_path)
        