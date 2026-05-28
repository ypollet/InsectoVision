#Constants
import os

DEFAULT_LABEL = "Insect"
DEFAULT_MODEL = os.path.join("model","final_23.pt")

DEFAULT_CONF = 0.85

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
PADX = 0 #x axis padding between buttons/labels
PAD_BOX = 5


MIN_SIZE_CANVAS = 500
WIDTH_LINE = 3
RADIUS_CIRCLE = 4
FONT_SCALE = 8e-3
