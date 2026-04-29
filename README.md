# InsectoVision
Welcome to the InsectoVision github repository !

As of June 2025, this project provides the state-of-the-art for insect detection in entomological boxes.
This work provides foundational tools for a complete and automatic processing of digital entomological collections.
Specifically, it allows accurate detection and localization of individual insects in entomological boxes, and does
so by carefully fine-tuning YOLOv8s on a tiny training set of 139 images (not provided in the project). Moreover, 
we pave the way for future dataset enlargement with the implementation of various Active Learning (AL) techniques.

Therefore, the project is ideal both for entomologists and for programmers who wish to optimally fine-tune YOLO 
on their very small single-class custom dataset : by using dynamic multi-phase fine-tuning, and by incrementally 
enlarging your labeled dataset using our AL sampling strategies, you will be able to reach good performance with 
very few training instances.

The project is completely open-source, and every step from pre-processing to training and inference is detailed and
commented.

## Tools overview
### Box Annotation UI

This script provides a graphical user interface that gives access to part of the functionality the inferece tools
without the need to use a terminal. Images can be viewed, scanned using an existing model and hand annotated.
From annotated images, a summary of specimens can be created as well as individual crops of each.

### Inference tool

Given a folder of entomological boxes' images, this script runs our detector's inference on each of them and
stores the YOLO format predictions in an 'output' folder. Several parameters can be chosen to choose the desired
inference pipeline.

### Training tool

Given a YOLO-format dataset (with an 'images' and a 'labels' subfolder), this script runs training using dynamic
multi-phase fine-tuning on YOLOv8s (or any other model initialization given as parameter). All training hyper-parameters 
can be chosen freely using command-line arguments. Resulting models are stored into 'output.pt' and
'output.keras'.

### Memory analysis tool

Given your available hardware, and more specifically your GPU memory, this tool will perform grid-search on every
YOLO model size, and decide what is the optimal training image resolution for each of them. For example, 
the training configuration of our detector 'final_23.pt' was chosen using this script : we found that training 
YOLOv8s on 640*640 images would use 6.2GB. Given our 8GB RAM, this configuration optimally leveraged our
hardware. The results of the analysis are printed into STDOUT.

### AL selection tool

Given an unlabeled pool of images, a detection model and its labeled training set, this tool selects an optimal
sample of the unlabeled pool, to be merged into the training set for subsequent training. Selection is made
according to an AL strategy given as command-line argument.

### AL re-training tool

Given a YOLO-format labeled dataset, the model trained on that dataset, and a new selection of labeled 
images (supposedly selected with AL selection), this tool merges the new set into the original training set and
performs fine-tuning of the model on that new enlarged dataset.

### Performance analysis tool

Given images, ground-truth labels, and model predictions, this tool computes the average precision, recall,
f1-score, map50 and map50-95 per image, and prints the results into STDOUT.

## Use

Note that for more precise information, all scripts are runnable with the command-line argument '-h' or '--help', 
which will explain the usage of all arguments in more details.

### Box annotation UI

In the "File" tab:
 -"Select image folder": opens a file navigator and lets the user select the working directory, either one already
                        worked on or one containing images. In the latter case, the folders "images" "labels" and 
                        "raw_ai_output" will be created.
 -"Create folder from URL list": lets the user select a .txt file with URLs containing images. Those images will be
                                 downloaded and put into a folder like in the previous case.
 -"Open selected images": loads the working directory's images and annotations (if any) to the visual interface.
 -"Scan selected images": runs the inference pipeline on the working directory images and stores the results in
                          "raw_ai_output".
 -"Quick open" and "Quick open from URL": selects (from an image directory or URL list), loads and runs inference
                                          (if "raw_ai_output" and "labels" do not contain a file for every image).
 -"Summarize saved boxes": creates a "summary.csv" file that contains the type and amount of each class of insect
                           in the saved boxes, as well as the total amount.
 -"Crop specimens from current box": creates a "crops" directory and saves in it a croped image of each specimen
                                    and paper tag with a bounding box.
 -"Parameters": allows to choose the detection model and wether to enable the post detection classifier

With an image open:
 -Left click on a bounding box: selects the bounding box (shown in blue)
 -Shift + left click on a bounding box: adds the bounding box to current selection 
 -Hold left click and drag: selects multiple bounding boxes
 -"Previous" and "Next": allow to browse the set of images
 -"Good detection" and "Bad detection": mark the currently selected boxes as correct/incorrect detection
 -"New box": adds a new specimen bounding box 
 -"Combine boxes": combines together all selected bounding boxes
 -"Add label": adds a label to the selected bounding boxes
 -"Save": saves the current image's bounding boxes to the "labels" directory
 -"Confidence threshold" slider: changes the confidence threshold. Bounding boxes with confidence above it are displayed
                                 in green and those below it, in yellow

In the "Edit" tab:
 -"New specimen box": same as "New box" in interface
 -"New tag box": adds a new bounding box meant for paper tags. It is saved separately and isn't used in training, detection
                 or selection
 -"Combine selected boxes": same as "Combine boxes" in interface
 -"Add label": same as "Add label" in interface

### Inference tool

The input images folder specified in '--input_folder' must contain images in jpg format. 
Typical inference can be run with the following command :<br>
`python inference_pipeline.py --input_folder my_images`<br><br>
Several arguments can be added for a more controllable inference pipeline, and we list some of them here :
 - write_conf : this argument is recommended if you want confidence level information into your txt
                output label files. Corresponding confidence levels will be written at the end of each 
                line. This information is essential for the UI as well as for the performance analysis tool.
 - model : the custom .pt detector you wish to use (default : model/final_23.pt).
 - classifier : the custom .keras post-detection binary classifier you wish to use (default : model/final_23.keras).
 - img_size : if you use a custom model trained on a resolution different from 640*640, you should specify image
              size here (default : 640).
 - detection_only : this skips the costly post-detection filtering made by the binary classifier, enabling
                    much faster inference with slightly reduced performances.
 - silent : if you do not want to be notified at directory deletions. Specifically, any already present 
            'output' folder will be completely replaced without any warning.

### Training tool

The input dataset in '--dataset' must be in standard YOLO format (with subfolders 'images' and 'labels').
Annotation files follow the YOLO format. Every line of the txt files corresponds to one bounding box,
of the form :<br>
`<class_id> <x_center> <y_center> <width> <height>`<br><br>
Typical training can be performed with the following command :<br>
`python training_pipeline.py --dataset my_yolo_format_dataset`<br><br>
Resulting detectors and post-detection binary classifiers will be stored into 'output.pt' and 'output.keras'
respectively.<br><br>
Note that the former command will only work for macos. If running on windows or linux, specify '--gpu 0'.
The chosen value will be put into the '--device' argument of YOLO training. More information at this link :
https://docs.ultralytics.com/modes/train/ <br><br>
Several arguments can be added for a more controllable training pipeline, and we list some of them here :
- fine_tuning_steps : number of steps for dynamic multi-phase fine-tuning. We advise a minimum of 3,
                      and it can be at maximum 23, because there are only 22 model layers to unfreeze (default : 23). 
- lr0 : the initial learning rate at the first fine-tuning step (default: 0.01).
- batch_init : initial batch size (default: 16).
- batch_min : minimum batch size (default: 8). If you do not want batch size decreasing, choose
              batch_min = batch_init.
- epochs : number of epochs at every fine-tuning step (default: 20)
- patience : patience at every step (default: 5)
- tp_ratio : the true positive ratio you wish in the training set of your binary post-detection classifier.
             Default is 0.8, because it was found that training on a balanced set failed to converge.
- model : weight initialization for detector's training (default: yolov8s.pt)
- detection_only : skips the post-detection classifier's training.
- classification_only : skips the detector's training, and only trains the classifier using the predictions
                        of the detector specified in '--model'.
- replace_all : if you do not want to be notified at directory/file deletions. Specifically, any already 
                present 'classify' folder and 'output.pt', 'output.keras' files will be completely replaced 
                without any warning.

### Memory analysis tool

This tool can be used in two ways : (1) find the optimal training configuration given your GPU memory available
and (2) compute the theoretical memory usage of a given training configuration.

1. Optimal configuration. This option must be set with the '--find_opti' argument. Two additional arguments are
   available for this option :
    - ram : total gpu memory in GBytes.
    - system_ram : unusable gpu memory because of system allocations or any source of overhead (default : 1.5GB)<br>
   Typical run command : `python memory_analysis.py --find_opti --ram 8`

2. Memory usage computation. This is the default option. Given a YOLO model size (--model_size), image resolution
   (--img_size), and batch size (--batch), it will compute memory usage per image and per batch. Model size can be 
   'n', 's', 'm', 'l' or 'x'. Two additional arguments are available :
    - fp : floating point precision (default : 32)
    - nc : number of classes (default : 1)<br>
   Typical run command : `python memory_analysis --model_size x --img_size 2016 --batch 16`

### AL selection tool

Given a labeled dataset (--labeled) in YOLO format, a model (--model) supposedly trained on that dataset, an
unlabeled pool (--unlabeled) with an 'images' subfolder, and a subsampling strategy (--strategy), this tool
samples a selection of '--size' images from the unlabeled pool for subsequent AL retraining. Several strategies
are available :<br>
`['random', 'uniform', 'uncertainty', 'diversity', 'supervised']`<br>
Note that the two first strategies do not necessitate a specified '--model' or '--labeled', and the third one
does not need '--labeled'. However, 'diversity' and 'supervised' need all three.<br><br>
AL selection can be run with the following typical run command :<br>
`python active_selection.py --model my_model.pt --unlabeled my_pool --strategy uncertainty --size 50`<br><br>
The output selection will be stored into 'selection' and the remaining ones will be put into 'unlabeled'. We advise
that you replace your unlabeled pool with this remaining pool.

### AL retraining tool

Given a labeled dataset (--labeled) in YOLO format, a model (--model) supposedly trained on that dataset, and a 
new set of labeled images (--new_images) supposedly selected using the AL selection tool, this tool merges the
new images into the labeled dataset (using validation-replacement strategy) and then fine-tunes the given model
on the new enlarged dataset. To do so, it trains the network with the given model as weight initialization, 
and uses a smaller learning rate, initial batch size, number of fine-tuning steps and number of epochs.
Specifically, given the relative size of the new selection, it will reduce '--original_lr0', '--original_batch',
'--original_nb_steps' and '--original_epochs' according to a formula detailed in the project paper. All those
'original' arguments should be the original training hyper-parameters of the given model.<br><br>
A typical run command can be found below :<br>
`python retrain.py --labeled my_original_dataset --model my_model.pt --new_images my_new_labeled_images`<br><br>
Resulting models will be stored into 'new_model.pt' and 'new_model.keras'. The original dataset will now contain the
new labeled images.<br><br>
Several additional arguments can be specified, and we list some of them below :
 - gpu : depends on your OS and on the number of gpu cores you have available. The specified value will be passed
         to the '--device' argument of YOLO training, more info on https://docs.ultralytics.com/modes/train/.
         (default : mps)
 - detection_only : skips classification retraining.
 - classification_only : skips detection retraining.

### Best practice for Active Learning

If you wish to optimally enlarge your dataset using active learning, follow those steps :
1. Train a model on your current labeled dataset using the training tool (should take several hours).
2. Select a sample from your unlabeled pool using the AL selection tool.
3. Replace your unlabeled pool with the 'unlabeled' directory produced by step 2.
4. Annotate images in 'selection' (produced by step 2) using the box annotation UI, ideally
   using the model produced by step 1.
5. Enlarge your dataset and fine-tune your model using the AL retraining tool (should take less than
   and hour).
6. Using the model produced by step 5, repeat steps 2 to 5 until you are happy with performances, or you
   exceeded your annotation budget.

### Performance analysis tool

Given folders '--images' of jpg images, '--ground_truth' of ground_truth YOLO labels, and '--predictions' filled
with predictions made by the model which performance you wish to measure, this tool will print in command line
the average precision, recall, f1-score, map50 and map50-95 per image.<br><br>
A typical run command can be found below :<br>
`python performance.py --images my_test_images --predictions my_model_predictions --ground_truth
my_test_labels`<br><br>
Several additional arguments can be specified, and we list some of them below :
 - min_conf : minimum confidence level for predictions to be taken into account (default : 0).
 - max_overlap : IoU overlap threshold. If the overlap between two boxes exceeds that threshold, non-maximum 
                 suppression will be performed based on confidence level (default : 1, which means no NMS).