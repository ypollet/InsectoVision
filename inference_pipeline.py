import copy
import shutil
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
    # also silence the prompt for internal recursive calls on a "tile" subfolder (see args.tiling below)
    

    # Force detection-only mode if high-precision is enabled,
    # because high-precision already uses the classifier's predictions in its input saliency maps,
    # so false-positive filtering is not needed
    if args.high_precision:
        args.detection_only = True

    times = [] # For tracking inference times

    # Loop through all images in the input folder
    list_images = []
    if os.path.isdir(args.input):
        input_folder = args.input
        list_images = [f"{input_folder}/{x}" for x in sorted(os.listdir(args.input))]
        api.warn_user_if_directory_exists(args.output, silent=args.silent or args.input.endswith("tile"))
    elif os.path.isfile(args.input):
        input_folder = os.path.dirname(os.path.realpath(args.input))
        list_images = [os.path.realpath(args.input)]
    # else input doesn't exist
    
    # TODO : error with input and input folder
    # TODO : log images that are tiles
    list_labels = []
    for image_path in list_images:
        image_file = os.path.basename(image_path)
        # skip non-image entries, e.g. a "tile"/"output" subfolder left over from a recursive tiling call
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        start = time.time() # Start timing inference
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
            
            results = model.predict(source=image, conf=args.conf, imgsz=args.img_size, iou=args.max_iou,
                                    max_det=1000, verbose=args.verbose)
            pred = api.store_predictions(results)
            pred = [api.yolo_to_bbox(x, image_size[0], image_size[1]) for x in pred]
            pred = [x for x in pred if x[-1] > args.conf]

            # Apply optional non-ML filtering
            if not args.no_filtering:
                pred = api.remove_overlapping_regions(pred)
                #pred = api.filter_bboxes_zscore(pred)
            pred_list.extend(pred)

        # Final overlap filtering across ensemble
        if not args.no_filtering:
            pred_list = api.remove_overlapping_regions(pred_list, args.max_overlap)

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

        output = os.path.join(args.output, image_file[:-4] + ".txt")
        list_labels.append(output)
        # Save predictions in YOLO format, with additional confidence level if write_conf is enabled
        api.save_yolo_format(pred_list, image_size,
                             output, write_conf=args.write_conf)

        end = time.time()
        if not args.silent:
            print(f"Time elapsed: {end - start:.4f} seconds")
        times.append(end - start)

    # Print average inference time
    if not args.silent:
        print("Average inference time on all images:", np.mean(np.asarray(times)))

    if args.tiling:
        # Recursive tiling: images whose mean detected-bbox area is too small (relative to the
        # full image) are re-run through this same pipeline on 4 higher-resolution corner crops
        # instead, and the crops' detections are merged back into the original image's
        # coordinates. This block can recurse more than one level deep (tiling a tile again) if a
        # crop is still too coarse after one round.
        smaller_insects_indices = api.select_smaller_insect_boxes(list_labels, bbox_area_threshold=args.min_bbox)

        if len(smaller_insects_indices) == 0:
            # Nothing (left) to tile here. If this call is itself processing a "tile" folder (we
            # recursed at least once to get here), merge its tile-level detections back into the
            # parent image's coordinates and hand the merged folder back up to the caller.
            if input_folder.endswith("tile"):
                if not args.silent:
                    print("Merging tiles...")
                return api.merge_tiles(input_folder, args.output)
            # Otherwise (top-level call, nothing needed tiling): nothing more to do.
        else:
            # Some images have detections that are too small: tile just those, then re-run
            # detection on the tiles.
            api.warn_user_if_directory_exists(os.path.join(input_folder, "tile"), silent=True, make_dir=False)
            #images, labels = api.get_images_and_labels(input_folder, args.output)
            selected_images = [list_images[i] for i in range(len(list_images)) if i in smaller_insects_indices]
            selected_labels = [list_labels[i] for i in range(len(list_labels)) if i in smaller_insects_indices]
            if not args.silent:
                print(f"{len(selected_images)} images/tiles have small insects, tiling...")
            api.tile(selected_images, selected_labels, input_folder, label_folder=args.output, silent=args.silent)

            # "output" is shared/global and is about to be overwritten by the recursive call
            # below, so back up the current (full) detection results before that happens.
            api.warn_user_if_directory_exists(os.path.join(input_folder, "output"), silent=True, make_dir=True)
            api.copy_folder(args.output, os.path.join(input_folder, "output"))

            # Recurse on the freshly created tile folder. write_conf is forced on because
            # merge_tiles() needs confidence to deduplicate detections where tiles overlap.
            args_copy = copy.copy(vars(args))
            args_copy['input'] = os.path.join(input_folder, "tile")
            args_copy['output'] = os.path.join(input_folder, "output")
            args_copy['write_conf'] = True
            args_copy = argparse.Namespace(**args_copy)
            if not args.silent:
                print("Inferring...")
            merged_tile_labels = main(args_copy)

            # Complete the merged tile-based results (which only cover the small-bbox images)
            # with the untouched original detections of the images that didn't need tiling.
            api.update_labels(merged_tile_labels, os.path.join(input_folder, "output"))

            if input_folder.endswith("tile"):
                # We're inside a recursive call ourselves: our own images are tiles of a parent
                # image, so merge them one level further up before returning.
                if not args.silent:
                    print("Merging tiles...")
                return api.merge_tiles(input_folder, args.output)
            else:
                # Top-level call: the backup has been folded back into "output", clean it up.
                shutil.rmtree(os.path.join(input_folder, "output"))
                # merge_tiles() always writes confidence; strip it back out if the caller didn't
                # actually ask for it in the final output.
                if not args.write_conf:
                    api.remove_conf_column_folder("output")


def parse_args():
    parser = argparse.ArgumentParser(description="python inference_pipeline.py --input my_image_folder")

    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Confidence threshold (default: 0.01)"
    )
    parser.add_argument(
            "--max_iou",
            type=float,
            default=0.5,
            help="Maximum iou between detections (default: 1 - iou allowed)"
        )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=0.85,
        help="Maximum overlap between detections (default: 1 - overlap allowed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("model", "rbins.pt"),
        help="Path to detection model (default: rbins.pt, in the model directory)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=os.path.join("model", "rbins.keras"),
        help="Path to posterior classifier (default: rbins.keras, in the model directory)"
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
        "--min_bbox",
        type=float,
        default=0.001,
        help="Minimum mean bbox area ratio relative to total image. When less, "
             "tiling occurs to increase its size (default: 0.001)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input folder or file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Path to the output folder"
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
    parser.add_argument(
        "--tiling",
        action="store_true",
        help="Zooms in for small insects."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)