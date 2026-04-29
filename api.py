import os
import shutil

import cv2
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import numpy as np
import random
import yaml

seed = 69


def store_predictions(results):
    """
    Extracts normalized YOLO predictions from model results.

    Args:
        results: List of YOLO result objects, each with .boxes containing predicted bounding boxes.

    Returns:
        List of tuples (x_center, y_center, width, height, confidence), all normalized between 0 and 1.
    """
    predictions = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, w, h = box.xywhn[0].tolist()     # normalized coordinates
            conf = box.conf[0].item()              # confidence
            predictions.append((x, y, w, h, conf))

    return predictions


def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """
    Converts a bounding box from YOLO format ([+ confidence])
    to absolute-coordinate (x_min, y_min, x_max, y_max, confidence) format.

    Args:
        yolo_bbox: Tuple of normalized (x_center, y_center, width, height, [confidence]).
        img_width: Width of the image.
        img_height: Height of the image.

    Returns:
        Bounding box as absolut-coordinate [x_min, y_min, x_max, y_max, confidence], which is the custom format.
        If input yolo_bbox does not have field confidence, it will be set to 1.
    """

    if len(yolo_bbox) == 4:
        yolo_bbox = yolo_bbox + (1,)
    x_center, y_center, width, height, conf = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    x_max = (x_center + width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    y_max = (y_center + height / 2) * img_height
    return [x_min, y_min, x_max, y_max, conf]


def compute_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU value (float).
    """
    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def is_largely_contained(box1, box2, threshold=0.5):
    """
    Checks if box1 is largely contained within box2 by intersection area ratio,
    or inversely if box2 is largely contained in box1.

    Args:
        box1: First bounding box.
        box2: Second bounding box.
        threshold: Intersection area ratio to be considered 'contained'.

    Returns:
        True if the containment ratio exceeds threshold,
        in any of the two ways (box1-in-box2 or box2-in-box1)
    """

    if box1 is None or box2 is None:
        return False

    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # when one of the boxes' area is close to the intersection's area,
    # it means that this box is largely contained in the other one.
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    if box1_area == 0 or box2_area == 0:
        return True
    return inter_area/box1_area > threshold or inter_area/box2_area > threshold


# LABEL_BIAS allows containment as valid match, by default.
# This is due to an initial bias in the labelling.
LABEL_BIAS = True
def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5, iosa_threshold=0.7, label_bias=LABEL_BIAS):
    """
    Evaluates detection performance by comparing predicted and ground truth boxes.

    Args:
        pred_boxes: List of predicted boxes with confidence (custom format).
        gt_boxes: List of ground truth boxes (custom format).
        iou_threshold: IoU threshold for matching.
        iosa_threshold: Intersection over smallest area threshold.
                        Additional containment threshold, only relevant if label_bias is true.
        label_bias: If True, allows containment match as valid match.

    Returns:
        Tuple: (false_positive_list, metrics_dict),
        with metrics containing entries precision, recall, f1_score and ap (average precision)
    """
    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return [], {"precision": 1, "recall": 0, "f1_score": 0, "ap": 0}
    elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return list(pred_boxes), {"precision": 0, "recall": 1, "f1_score": 0, "ap": 0}

    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    tps = []
    fps = []

    for pred in pred_boxes:
        best_iou = 0
        best_match = None
        best_box = None
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_match = i
                best_box = gt

        bias_condition = label_bias and is_largely_contained(best_box, pred, iosa_threshold)
        if (best_iou >= iou_threshold or bias_condition) and best_match not in matched_gt:
            tp += 1
            matched_gt.add(best_match)
            tps.append(1)
            fps.append(0)
        else:
            fp += 1
            tps.append(0)
            fps.append(1)

    fp_list = [pred_boxes[i] for i in range(len(pred_boxes)) if fps[i] == 1]
    fn = len(gt_boxes) - len(matched_gt)

    tps = np.cumsum(tps)
    fps = np.cumsum(fps)

    recalls = tps / len(gt_boxes)
    precisions = tps / (tps + fps + 1e-6)

    # Interpolate precision (COCO-style 101-point interpolation)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        ap += max(p) if p.size else 0
    ap /= 101

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return fp_list, {"precision": precision, "recall": recall, "f1_score": f1_score, "ap": ap}


def compute_map_50_95(pred_list, gt_list):
    """
    Computes mean average precision over IoU thresholds from 0.5 to 0.95.

    Args:
        pred_list: List of predicted boxes (custom format).
        gt_list: List of ground truth boxes (custom format).

    Returns:
        Mean Average Precision (float).
    """
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        _, metrics = evaluate_detections(pred_list, gt_list, iou_threshold=iou, label_bias=False)
        ap = metrics["ap"]
        aps.append(ap)
    return np.mean(aps)


def compute_map_50(pred_list, gt_list, label_bias=False):
    """
    Computes average precision at 0.5 IoU threshold.

    Args:
        pred_list: List of predicted boxes (custom format).
        gt_list: List of ground truth boxes (format).
        label_bias: Whether to allow relaxed matching.

    Returns:
        Average Precision at IoU=0.5.
    """
    _, metrics = evaluate_detections(pred_list, gt_list, iou_threshold=0.5, label_bias=label_bias)
    ap = metrics["ap"]
    return np.mean(ap)


def txt_to_tuple_list(file_name):
    """
    Converts YOLO-style .txt annotation file into list of tuples.

    Args:
        file_name: Path to .txt file.

    Returns:
        List of tuples containing bounding box data.
    """
    ret_list = []
    with open(file_name, mode='r', newline='') as file:
        for line in file.readlines():
            row = line.split()
            if row[0][0] <= '9':
                ret_list.append(tuple([float(x) for x in row[1:]]))
    return ret_list


def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2, show_conf=False):
    """
    Draws bounding boxes on an image using OpenCV.

    Args:
        image: OpenCV image array.
        bboxes: List of (x_min, y_min, x_max, y_max, conf).
        color: Color for bounding boxes (RGB).
        thickness: Line thickness.
        show_conf: If True, draws confidence score.
    """

    for (x_min, y_min, x_max, y_max, conf) in bboxes:
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

        if show_conf:

            label = f"{conf:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 4

            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            cv2.rectangle(image,
                          (int(x_min), int(y_min) - text_height - 8),
                          (int(x_min) + text_width, int(y_min)),
                          color,
                          cv2.FILLED)

            cv2.putText(image, label,
                        (int(x_min), int(y_min) - 6),
                        font, font_scale,
                        (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)


def remove_overlapping_regions(bboxes, overlap_treshold=0.5):
    """
    Removes overlapping bounding boxes based on containment logic.

    Args:
        bboxes: List of bounding boxes (custom format).
        overlap_treshold: Containment (iosa) threshold.

    Returns:
        Filtered list of bounding boxes.
    """
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    to_remove = np.zeros(len(bboxes))
    for i in range(len(bboxes)):
        if not to_remove[i] == 1:
            for j in range(i+1, len(bboxes)):
                if not to_remove[j] == 1 and is_largely_contained(bboxes[i], bboxes[j], threshold=overlap_treshold):
                    to_remove[j] = 1
    ret_list = [bboxes[i] for i in range(len(bboxes)) if to_remove[i] == 0]
    return ret_list


def remove_overlapping_regions_wrt_iou(bboxes, overlap_treshold=0.5):
    """
    Removes overlapping bounding boxes based on IoU threshold.

    Args:
        bboxes: List of bounding boxes.
        overlap_treshold: IoU threshold to determine overlap.

    Returns:
        Filtered list of bounding boxes.
    """
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    to_remove = np.zeros(len(bboxes))
    for i in range(len(bboxes)):
        if not to_remove[i] == 1:
            for j in range(i+1, len(bboxes)):
                if not to_remove[j] == 1 and compute_iou(bboxes[i], bboxes[j]) > overlap_treshold:
                    to_remove[j] = 1
    ret_list = [bboxes[i] for i in range(len(bboxes)) if to_remove[i] == 0]
    return ret_list


def filter_bboxes_zscore(bboxes, threshold=5):
    """
    Filters out bounding boxes with anomalous areas using z-score.

    Args:
        bboxes: List of bounding boxes (custom format).
        threshold: Z-score threshold.

    Returns:
        Filtered bounding boxes.
    """
    areas = np.array([(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max, _ in bboxes])

    mean_area = np.mean(areas)
    std_area = np.std(areas)

    filtered_bboxes = [
        b for b, area in zip(bboxes, areas) if abs((area - mean_area) / std_area) < threshold
    ]

    return filtered_bboxes


def save_regions(image, boxes, output_path, output_length, resize=None, resize_mode='bilinear'):
    """
    Saves cropped image regions based on bounding boxes.

    Args:
        image: Input numpy image array (RGB).
        boxes: List of bounding boxes (custom format).
        output_path: Directory to save cropped images.
        output_length: Starting index for naming.
        resize: Resize dimensions (height, width) if needed.
        resize_mode: 'bilinear' or 'pad'.

    Returns:
        Updated output_length after saving.
    """
    nb_discarded = 0
    for i, region in enumerate(boxes):
        x_min, y_min, x_max, y_max, _ = map(int, region)
        if x_min == x_max or y_min == y_max:
            nb_discarded += 1
            continue
        cropped_region = image[y_min:y_max, x_min:x_max]
        if resize is not None:
            if resize_mode == 'bilinear':
                cropped_region = tf.image.resize(cropped_region, resize, method='bilinear')
            elif resize_mode == 'pad':
                try:
                    cropped_region = tf.image.resize_with_pad(cropped_region, resize[0], resize[1])
                except tf.errors.InvalidArgumentError:
                    nb_discarded += 1
                    continue
            else:
                raise ValueError("Invalid resizing mode")
            cropped_region = np.asarray(cropped_region, dtype=np.uint8)
        cv2.imwrite(os.path.join(output_path, "i" + str(output_length + i - nb_discarded) + ".jpg"), cropped_region)
    return output_length + len(boxes) - nb_discarded


def save_yolo_format(bboxes, image_size, output_txt_path, class_id=0, write_conf=False):
    """
    Converting custom-format bounding boxes to YOLO-format and saving them to a file.

    Args:
        bboxes: List of boxes (x_min, y_min, x_max, y_max, conf) (custom-format).
        image_size: Tuple (width, height).
        output_txt_path: File path for output.
        class_id: Class ID for all boxes. (this method only saves single-class predictions)
        write_conf: If True, includes confidence value.

    Returns:
        None
    """

    img_width, img_height = image_size

    with open(output_txt_path, "w") as f:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf = bbox

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min

            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            line_to_write = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            if write_conf:
                line_to_write += f" {conf:.6f}"
            f.write(line_to_write + "\n")


def warn_user_if_directory_exists(dir, silent=False, make_dir=True):
    """
    Warns user before deleting an existing directory, then recreates it.

    Args:
        dir: Path to directory.
        silent: If True, does not prompt the user.
        make_dir: If True, creates the directory after deletion.

    Returns:
        None
    """
    if os.path.exists(dir):
        if not silent:
            ans = input(f"{dir} folder already exists, do you wish to replace (r) or cancel (c) ?\n")
            while not (ans == 'r' or ans == 'c'):
                input("Invalid response, choose between replace (r) or cancel (c)\n")
            if ans == 'c':
                exit()
            else:
                shutil.rmtree(dir)
        else:
            shutil.rmtree(dir)
    if make_dir:
        os.makedirs(dir)


def warn_user_if_file_exists(file, silent=False):
    """
    Warns user before overwriting an existing file.

    Args:
        file: File path.
        silent: If True, deletes without prompting.

    Returns:
        None
    """
    if os.path.exists(file):
        if not silent:
            ans = input(f"{file} file already exists, do you wish to replace (r) or cancel (c) ?\n")
            while not (ans == 'r' or ans == 'c'):
                input("Invalid response, choose between replace (r) or cancel (c)\n")
            if ans == 'c':
                exit()
            else:
                os.remove(file)
        else:
            os.remove(file)


def build_fcnn():
    """
    Builds a fully convolutional neural network using EfficientNetB0 as backbone.

    Returns:
        Keras model object (FCNN).
    """
    convnet = tfk.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet"
    )
    convnet.trainable = False

    inputs = tfk.Input(shape=(None, None, 3))
    x = convnet(inputs, training=False)

    x = tfkl.Conv2D(filters=256, kernel_size=1, activation='gelu', name='conv1')(x)
    x = tfkl.Dropout(0.2, name='conv1_dropout')(x)

    x = tfkl.Conv2D(filters=128, kernel_size=1, activation='gelu', name='conv2')(x)
    x = tfkl.Dropout(0.2, name='conv2_dropout')(x)

    outputs = tfkl.Conv2D(filters=2, kernel_size=1, activation='softmax', name='output_conv')(x)

    model = tfk.Model(inputs=inputs, outputs=outputs, name='fcnn_model')
    return model


def transfer_dense_to_conv(dense_layer, conv_layer):
    """
    Transfers weights from a dense layer to a convolutional layer.

    Args:
        dense_layer: Keras Dense layer.
        conv_layer: Keras Conv2D layer.

    Returns:
        None
    """
    dense_weights, dense_bias = dense_layer.get_weights()
    conv_weights = np.expand_dims(np.expand_dims(dense_weights, axis=0), axis=0)  # Reshape for Conv2D
    conv_layer.set_weights([conv_weights, dense_bias])


def make_fcnn(classifier):
    """
    Loads a dense classifier and converts it to a FCNN.

    Args:
        classifier: Path to trained classifier model.

    Returns:
        Keras FCNN model.
    """
    fcnn = build_fcnn()
    extractor = tfk.models.load_model(classifier)
    for layer_fcnn, layer_orig in zip(fcnn.layers[1].layers, extractor.layers[1].layers):
        layer_fcnn.set_weights(layer_orig.get_weights())

    transfer_dense_to_conv(extractor.get_layer("dense1"), fcnn.get_layer("conv1"))
    transfer_dense_to_conv(extractor.get_layer("dense2"), fcnn.get_layer("conv2"))
    transfer_dense_to_conv(extractor.get_layer("output"), fcnn.get_layer("output_conv"))

    return fcnn


def has_corresponding_image(image_folder, label):
    """
    Checks whether an image file corresponding to a label exists in a folder.

    Args:
        image_folder: Directory containing images.
        label: Filename of label (e.g., .txt or annotation).

    Returns:
        True if image exists, else False.
    """
    for i in os.listdir(image_folder):
        if i.startswith(label[:-4]):
            return True
    return False


def get_images_and_labels(image_folder, label_folder):
    """
    Retrieves image files from a specified directory and their corresponding label files from another directory.

    Args:
        image_folder (str): Path to the directory containing image files.
        label_folder (str): Path to the directory containing label files.

    Returns:
        tuple: A tuple containing two sorted and one-to-one matching lists:
            - A list of image file names that have corresponding labels.
            - A list of label file names with a matching image.
    """
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(image_extensions) and
           os.path.exists(os.path.join(label_folder, os.path.splitext(f)[0] + ".txt"))
    ]
    label_files = [f for f in os.listdir(label_folder)
                   if f.endswith(".txt") and
                   has_corresponding_image(image_folder, f)]
    image_files.sort()
    label_files.sort()

    return image_files, label_files


def get_images(image_folder):
    """
    Retrieves image files from a specified directory.

    Args:
        image_folder (str): Path to the directory containing image files.

    Returns:
        A list of image file names that have image extensions (jpg, jpeg, png)
    """
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(image_extensions)
    ]
    image_files.sort()

    return image_files


def make_set_from_indices(name, images_dir, labels_dir, indices, silent=True):
    """
    Creates a new dataset by copying specific images and labels (those specified in a list of indices)
    from the source directories to a new set of directories.

    Args:
        name (str): The name of the new dataset, used to create subdirectories.
        images_dir (str): Path to the source directory containing image files.
        labels_dir (str): Path to the source directory containing label files.
        indices (list[int]): A list of indices indicating which images and labels to copy.
        silent (bool): Whether or not to suppress warnings if the directory already exists.

    Returns:
        None
    """
    warn_user_if_directory_exists(name, silent=silent)
    test_images = os.path.join(name, "images")
    test_labels = os.path.join(name, "labels")
    os.makedirs(test_images)
    os.makedirs(test_labels)

    images_with_labels, labels = get_images_and_labels(images_dir, labels_dir)
    all_images = get_images(images_dir)
    nb_discarded = 0

    for i in range(len(all_images)):
        if all_images[i] in images_with_labels:
            if i in indices:
                shutil.copy2(os.path.join(images_dir, all_images[i]),
                             os.path.join(test_images, all_images[i]))
                shutil.copy2(os.path.join(labels_dir, labels[i - nb_discarded]),
                             os.path.join(test_labels, labels[i - nb_discarded]))
        else:
            nb_discarded += 1


def make_selection_from_indices(name, images_dir, indices, silent=True):
    """
    Creates a new selection by copying specific images (those specified in a list of indices)
    from the source directory to a new directory.

    Args:
        name (str): The name of the new dataset directory.
        images_dir (str): Path to the source directory containing image files.
        indices (list[int]): A list of indices indicating which images and labels to copy.
        silent (bool): Whether or not to suppress warnings if the directory already exists.

    Returns:
        None
    """
    warn_user_if_directory_exists(name, silent=silent)
    test_images = os.path.join(name, "images")
    os.makedirs(test_images)

    images = get_images(images_dir)

    for i in indices:
        shutil.copy2(os.path.join(images_dir, images[i]),
                     os.path.join(test_images, images[i]))


def show_images(images_dir, images, indices=None, pred_dir=None):
    """
    Displays images from a directory and optionally overlays bounding boxes for predicted labels.

    Args:
        images_dir (str): Path to the directory containing image files.
        images (list[str]): A list of image filenames to display.
        indices (list[int], optional): A list of indices specifying which images to display. Defaults to None (all images).
        pred_dir (str, optional): Path to a directory containing prediction files for each image.
                                  Defaults to None (no bounding boxes will be drawn).

    Returns:
        None
    """
    if indices is None:
        indices = list(range(len(images)))
    for idx in indices:
        i = images[idx]
        image = cv2.imread(os.path.join(images_dir, i))
        if pred_dir is not None:
            pred_file = os.path.join(pred_dir, i[:-4] + ".txt")
            pred_list = txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
            if len(pred_list) > 0 and len(pred_list[0]) == 4:
                pred_list = [x + (1,) for x in pred_list]
            pred_list = [yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]
            draw_bboxes(image, pred_list, color=(255, 0, 0), thickness=5, show_conf=True)
        cv2.imshow(i, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def copy_folder(src_folder, dst_folder):
    """
    Copies all regular files from one folder to another.

    Args:
        src_folder (str): Path to the source directory containing files to copy.
        dst_folder (str): Path to the destination directory where files will be copied.

    Returns:
        None
    """
    os.makedirs(dst_folder, exist_ok=True)
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)


def shuffle_data(images, labels, seed):
    """
    Shuffles images and labels in unison using a fixed seed.

    Args:
        images (list): A list of image data or identifiers.
        labels (list): A list of corresponding labels.
        seed (int): A random seed to ensure reproducible shuffling.

    Returns:
        tuple: Two lists — shuffled images and shuffled labels, with their relative order preserved.
    """
    random.seed(seed)

    shuffled_images = list(images)
    shuffled_labels = list(labels)
    shuffled_images.sort()
    shuffled_labels.sort()
    indices = list(range(len(shuffled_images)))
    random.shuffle(indices)
    shuffled_images = [shuffled_images[i] for i in indices]
    shuffled_labels = [shuffled_labels[i] for i in indices]

    return shuffled_images, shuffled_labels


def update_yaml_paths(yaml_path, new_dataset):
    """
    Updates the train and val image paths in a YAML file and saves it to the new dataset directory.

    Args:
        yaml_path (str): Path to the original YAML configuration file.
        new_dataset (str): Path to the new dataset root directory.

    Returns:
        None
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    abs_new_dataset = os.path.abspath(new_dataset)
    data['train'] = os.path.join(abs_new_dataset, 'train', 'images')
    data['val'] = os.path.join(abs_new_dataset, 'val', 'images')

    output_yaml_path = os.path.join(new_dataset, os.path.basename(yaml_path))

    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Updated YAML saved to {output_yaml_path}")


def merge_datasets(original_dataset_dir, new_dir, output_dir, new_split_ratio=0.8, seed=seed, replace_val=True):
    """
    Merges a new dataset into an existing dataset, adjusting the train/val split according to the merging strategy specified by replace_val,
    and writes the result to a new output directory.

    Args:
        original_dataset_dir (str): Path to the original dataset root directory.
        new_dir (str): Path to the new dataset to merge into the original dataset.
        output_dir (str): Path to the directory where the merged dataset will be saved.
        new_split_ratio (float, optional): Proportion of new data to add to the training set.
                                           The rest goes to validation. Defaults to 0.8.
                                           Note that changing validation ratio from previous AL iterations
                                           may result in unexpected behavior, so it will produce an error
        seed (int, optional): Random seed for reproducibility.
        replace_val (bool, optional): Whether to replace part of the original validation set,
                                      with replaced data moved to training (validation replacement strategy).
                                      Setting to false gives the standard validation split strategy.
                                      Defaults to True.

    Returns:
        tuple: (S, k) where S is the original total train+val sample count, and k is the number
               of new samples from `new_dir`.
    """
    random.seed(seed)

    src_images = os.path.join(original_dataset_dir, "images")
    src_labels = os.path.join(original_dataset_dir, "labels")
    train_path_src_images = os.path.join(original_dataset_dir, "train", "images")
    train_path_src_labels = os.path.join(original_dataset_dir, "train", "labels")
    val_path_src_images = os.path.join(original_dataset_dir, "val", "images")
    val_path_src_labels = os.path.join(original_dataset_dir, "val", "labels")
    yaml_src = os.path.join(original_dataset_dir, "data.yaml")

    dst_images = os.path.join(output_dir, "images")
    dst_labels = os.path.join(output_dir, "labels")
    train_path_dst_images = os.path.join(output_dir, "train", "images")
    train_path_dst_labels = os.path.join(output_dir, "train", "labels")
    val_path_dst_images = os.path.join(output_dir, "val", "images")
    val_path_dst_labels = os.path.join(output_dir, "val", "labels")

    new_images_dir = os.path.join(new_dir, "images")
    new_labels_dir = os.path.join(new_dir, "labels")

    # Copy all existing and new data into the destination folder
    copy_folder(src_images, dst_images)
    copy_folder(src_labels, dst_labels)
    copy_folder(new_images_dir, dst_images)
    copy_folder(new_labels_dir, dst_labels)
    copy_folder(train_path_src_images, train_path_dst_images)
    copy_folder(train_path_src_labels, train_path_dst_labels)
    os.makedirs(val_path_dst_images, exist_ok=True)
    os.makedirs(val_path_dst_labels, exist_ok=True)

    # Update and write the new YAML
    update_yaml_paths(yaml_src, output_dir)

    # Compute size of the original training and validation sets
    train_length_src = len(os.listdir(train_path_src_images))
    val_length_src = len(os.listdir(val_path_src_images))
    S = train_length_src + val_length_src

    # Get new image/label pairs
    new_images, new_labels = get_images_and_labels(new_images_dir, new_labels_dir)
    k = len(new_labels)

    # Shuffle data for random selections
    images_to_add, labels_to_add = shuffle_data(new_images, new_labels, seed)
    current_val_images, current_val_labels = get_images_and_labels(val_path_src_images, val_path_src_labels)
    current_val_images, current_val_labels = shuffle_data(current_val_images, current_val_labels, seed)

    # Determine how much of the new data should go to val
    new_val_size = round((1 - new_split_ratio) * (S + k))
    dif = new_val_size - val_length_src
    k_val = dif if not replace_val else new_val_size
    k_val = min(k_val, k)

    # Determine how many current val samples to move to train (if replacing)
    M = round(new_split_ratio * k) if round(new_split_ratio * k) < len(current_val_images) \
        else len(current_val_images)

    # Raise error if new split ratio is different from the one of previous iteration (dif > 0.1)
    if abs((val_length_src / S) - (1 - new_split_ratio)) > 0.1:
        raise ValueError(f"New split ratio {new_split_ratio} is too dissimilar from the previous split ratio {1 - (val_length_src / S)},"
                         f"which may lead to unexpected behavior. Please specify a split ratio closer to {1 - (val_length_src / S)}.")

    # Copy new validation data
    for i in range(k_val):
        image_name = images_to_add[i]
        label_name = labels_to_add[i]
        shutil.copy2(os.path.join(new_images_dir, image_name),
                     os.path.join(val_path_dst_images, image_name))
        shutil.copy2(os.path.join(new_labels_dir, label_name),
                     os.path.join(val_path_dst_labels, label_name))

    # Optionally move some original val data to train and preserve rest in val
    if replace_val:
        for i in range(M):
            image_to_replace = current_val_images[i]
            label_to_replace = current_val_labels[i]
            shutil.copy2(os.path.join(val_path_src_images, image_to_replace),
                         os.path.join(train_path_dst_images, image_to_replace))
            shutil.copy2(os.path.join(val_path_src_labels, label_to_replace),
                         os.path.join(train_path_dst_labels, label_to_replace))
        for i in range(M, len(current_val_images)):
            image_name = current_val_images[i]
            label_name = current_val_labels[i]
            shutil.copy2(os.path.join(val_path_src_images, image_name),
                         os.path.join(val_path_dst_images, image_name))
            shutil.copy2(os.path.join(val_path_src_labels, label_name),
                         os.path.join(val_path_dst_labels, label_name))

    # Copy remaining new data to train
    if k > k_val:
        for i in range(k_val, k):
            image_name = images_to_add[i]
            label_name = labels_to_add[i]
            shutil.copy2(os.path.join(new_images_dir, image_name),
                         os.path.join(train_path_dst_images, image_name))
            shutil.copy2(os.path.join(new_labels_dir, label_name),
                         os.path.join(train_path_dst_labels, label_name))

    # If not replacing, preserve original val set entirely
    if not replace_val:
        copy_folder(val_path_src_images, val_path_dst_images)
        copy_folder(val_path_src_labels, val_path_dst_labels)

    return S, k


def convert_valid_to_real_indices(indices, valid_images, all_images):
    """
    Converts indices from a subset of valid images to their corresponding indices in the full image list.

    Args:
        indices (list[int]): Indices referring to positions in `valid_images`.
        valid_images (list): Subset of images.
        all_images (list): Complete list of all images.

    Returns:
        list[int]: Indices of the selected valid images within `all_images`.
    """
    return [all_images.index(valid_images[i]) for i in indices]