import configparser
import os
from tkinter import StringVar, BooleanVar

from src.consts import DEFAULT_MODEL, DEFAULT_DATASET_DIR, DEFAULT_CLASSIFICATION, DEFAULT_OVERLAP, DEFAULT_IOU

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")

SECTION = "Default_params"


class Config:

    def __init__(self):
        self.model = DEFAULT_MODEL
        self.dataset = DEFAULT_DATASET_DIR
        self._classification = BooleanVar(value=DEFAULT_CLASSIFICATION)
        self._classification.trace_add(mode="write", callback=lambda var, idx, mode : self.save())
        self._max_overlap = StringVar(value=DEFAULT_OVERLAP)
        self._max_overlap.trace_add(mode="write", callback=lambda var, idx, mode : self.save())
        self._max_iou = StringVar(value=DEFAULT_IOU)
        self._max_iou.trace_add(mode="write", callback=lambda var, idx, mode : self.save())

        self.read_config(CONFIG_PATH)

    @property
    def classification(self):
        return self._classification.get()

    @property
    def max_overlap(self):
        return self._max_overlap.get()
    @property
    def max_iou(self):
        return self._max_iou.get()
        
    def read_config(self, config_path):
        parser = configparser.ConfigParser()
        if os.path.exists(config_path):
            parser.read(config_path)
            if parser.has_section(SECTION):
                self.model = parser.get(SECTION, "model", fallback=self.model)
                self.dataset = parser.get(SECTION, "dataset", fallback=self.dataset)
                print(list(parser[SECTION].keys()))
                if "classification" in parser[SECTION].keys():
                    print(parser.get(SECTION, "classification", fallback=self.classification))
                    print(self.classification)
                    self._classification.set(parser.get(SECTION, "classification", fallback=self.classification))
                if "max_overlap" in parser[SECTION]:
                    self._max_overlap.set(parser.get(SECTION, "max_overlap", fallback=self.max_overlap))
                if "max_iou" in parser[SECTION]:
                    self._max_iou.set(parser.get(SECTION, "max_iou", fallback=self.max_iou))
                self.save()

    def reset(self):
        self.model = DEFAULT_MODEL
        self.dataset = DEFAULT_DATASET_DIR
        self._classification.set(DEFAULT_CLASSIFICATION)
        self._max_overlap.set(DEFAULT_OVERLAP)
        self._max_iou.set(DEFAULT_IOU)
        self.save()

    def save(self):
        parser = configparser.ConfigParser()
        parser[SECTION] = {
            "model": self.model or "",
            "dataset": self.dataset or "",
            "classification" : str(self.classification) or "",
            "max_overlap" : self.max_overlap or "",
            "max_iou": self.max_iou or "",
            
        }
        with open(CONFIG_PATH, "w") as f:
            parser.write(f)
