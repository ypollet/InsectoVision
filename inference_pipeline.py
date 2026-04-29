import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import api
import tensorflow as tf
from tensorflow import keras as tfk
import time


def main(args):
    # Determine whether we're working with a single model file or a directory of models
    if args.model.endswith(".pt"):
        to_be_ensembled = [args.model]
    else:
        to_be_ensembled = [os.path.join(args.model, x) for x in os.listdir(args.model) if x.endswith(".pt")]
        if len(to_be_ensembled) == 0:
            raise ValueError("Directory to ensemble does not contain .pt models")
    to_be_ensembled = [YOLO(x) for x in to_be_ensembled]

    # Prepare output directory
    api.warn_user_if_directory_exists("output", silent=args.silent)

    # Force detection-only mode if high-precision is enabled,
    # because high-precision already uses the classifier's predictions in its input saliency maps,
    # so false-positive filtering is not needed
    if args.high_precision:
        args.detection_only = True

    times = [] # For tracking inference times

    # Loop through all images in the input folder
    for image_file in os.listdir(args.input_folder):

        start = time.time() # Start timing inference

        image_path = os.path.join(args.input_folder, image_file)
        image = Image.open(image_path)
        image_size = image.size

        # If high-precision mode is requested, use FCNN heatmap visualization
        if args.high_precision:
            img_array = np.array(image, dtype=np.float32)  # Convert to numpy array
            factor = np.max(img_array.shape) / 2016 if np.max(img_array.shape) > 2016 else 1
            img_array = tf.image.resize(img_array, (int(img_array.shape[0] / factor), int(img_array.shape[1] / factor)),
                                        method=tf.image.ResizeMethod.BILINEAR)
            fcnn = api.make_fcnn(args.classifier)
            preds = fcnn.predict(np.expand_dims(img_array, axis=0), verbose=args.verbose)
            preds = np.squeeze(preds)

            # Resize, normalize, and visualize the heatmap
            heatmap = tf.image.resize(np.expand_dims(preds[:, :, 1], axis=-1), (img_array.shape[0], img_array.shape[1]),
                                      method=tf.image.ResizeMethod.BILINEAR)
            heatmap = np.uint8(heatmap * 255)  # Scale to [0,255]
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
            heatmap_RGB = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            alpha = 0.1
            image = cv2.addWeighted(np.uint8(img_array), 1 - alpha, heatmap_RGB, alpha,0)  # Blend images

        # Read original image using OpenCV
        cv_image = cv2.imread(image_path)  # Load image
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pred_list = []

        # Run detection on each ensemble model
        for model in to_be_ensembled:
            results = model.predict(source=image, conf=args.conf, imgsz=args.img_size, iou=args.max_overlap,
                                    max_det=1000, verbose=args.verbose)
            pred = api.store_predictions(results)
            pred = [api.yolo_to_bbox(x, image_size[0], image_size[1]) for x in pred]
            pred = [x for x in pred if x[-1] > args.conf]

            # Apply optional non-ML filtering
            if not args.no_filtering:
                pred = api.remove_overlapping_regions(pred)
                pred = api.filter_bboxes_zscore(pred)
            pred_list.extend(pred)

        # Final overlap filtering across ensemble
        if not args.no_filtering:
            pred_list = api.remove_overlapping_regions(pred_list)

        old_list = list(pred_list) # Preserves list state before posterior classification

        # If classifier is enabled, and we’re not in detection-only mode
        if not args.detection_only and len(pred_list) > 0:
            classifier_name = args.classifier
            classifier = tfk.models.load_model(classifier_name)

            resized_pred_regions = []

            # Resize each region of interest for classifier input
            if args.resize_mode == "pad":
                for region in pred_list:
                    x_min, y_min, x_max, y_max, _ = map(int, region)
                    cropped_region = cv_image_rgb[y_min:y_max, x_min:x_max].astype(np.float32)
                    cropped_region = tf.image.resize_with_pad(cropped_region, 256, 256)
                    resized_pred_regions.append(cropped_region)
                resized_pred_regions = np.asarray(resized_pred_regions, dtype=np.float32)

            elif args.resize_mode == "bilinear":
                cv_image_rgb = cv_image_rgb.astype(np.float32)
                normalized_pred_list = []
                for region in pred_list:
                    x_min, y_min, x_max, y_max, _ = region
                    normalized_pred_list.append((y_min / image_size[1], x_min / image_size[0],
                                                y_max / image_size[1], x_max / image_size[0]))
                resized_pred_regions = tf.image.crop_and_resize(
                    image=tf.expand_dims(cv_image_rgb, axis=0),  # [1, H, W, C]
                    boxes=normalized_pred_list,  # [N, 4] in normalized coords [y1, x1, y2, x2]
                    box_indices=tf.zeros(len(normalized_pred_list), dtype=tf.int32),  # image index for each box
                    crop_size=(256, 256)
                )

            else:
                raise ValueError("Invalid resizing mode")

            # Run classification on the cropped regions
            predictions = np.argmax(classifier.predict(resized_pred_regions, verbose=args.verbose), axis=-1) if len(
                resized_pred_regions) > 0 else np.array([])
            indices = list(np.where(predictions == 1)[0]) if len(predictions) > 0 else []

            # Keep only positively classified regions
            pred_list = [old_list[i] for i in range(len(old_list)) if i in indices]

        # Save predictions in YOLO format, with additional confidence level if write_conf is enabled
        api.save_yolo_format(pred_list, image_size,
                             os.path.join("output", image_file[:-4] + ".txt"), write_conf=args.write_conf)

        end = time.time()
        if not args.silent:
            print(f"Time elapsed: {end - start:.4f} seconds")
        times.append(end - start)

    # Print average inference time
    if not args.silent:
        print("Average inference time on all images:", np.mean(np.asarray(times)))


def parse_args():
    parser = argparse.ArgumentParser(description="python inference_pipeline.py --input_folder my_image_folder")

    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Confidence threshold (default: 0.01)"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=0.25,
        help="Maximum overlap between detections (default: 0.25)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("model", "final_23.pt"),
        help="Path to detection model (default: final_23.pt, in the model directory)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=os.path.join("model", "final_23.keras"),
        help="Path to posterior classifier (default: final_23.keras, in the model directory)"
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="pad",
        help="Resizing mode that was used in the training of the classifier. If you can afford "
             "a bit more overhead, training with mode \'pad\' will give greater accuracy. "
             "(default: \'pad\', faster but less accurate alternative : \'bilinear\'.)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input folder"
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="Use the high-precision low-recall model"
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Use detector only, and no classifier"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--write_conf",
        action="store_true",
        help="Add confidence for each bounding box prediction in the output txt files"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Nothing printed in stdout"
    )
    parser.add_argument(
        "--no_filtering",
        action="store_true",
        help="Disables overlapping box NMS (based on IoSA threshold) and severe outliers suppression "
             "in terms of box area. Do not use together with an ensemble."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)