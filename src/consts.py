#Constants
import os
from enum import Flag, auto

DEFAULT_LABEL = "Insect"
DEFAULT_MODEL = os.path.join("model","rbins.pt")

DEFAULT_CONF = 0.85

#Drawing reasons
NEW_BBOX = 1
NEW_TAG = 2
SELECTING = 3

#Bbox status
class Status(Flag):
    SELECTED=auto()     #User has selected the bbox
    REJECTED=auto()     #User has confirmed the bbox is incorrect
    DOUBT=auto()        #AI's confidence is below threshold
    SURE=auto()         #AI's confidence is above threshold
    CONFIRMED=auto()    #User has confirmed the bbox is correct
    ACCEPTED = SURE | CONFIRMED
    NO_UPDATE = REJECTED | CONFIRMED

COLORS =    {
                Status.SURE:"chartreuse4",
                Status.DOUBT:"gold",
                Status.CONFIRMED:"green2",
                Status.REJECTED:"red",
                Status.SELECTED:"blue",
            }


BWIDTH = 15 #Button width
PADX = 0 #x axis padding between buttons/labels
PAD_BOX = 5


MIN_SIZE_CANVAS = 500
MAX_SIZE_ENTOBOX_LIST = 200
WIDTH_LINE = 3
RADIUS_CIRCLE = 4
FONT_SCALE = 8e-3
