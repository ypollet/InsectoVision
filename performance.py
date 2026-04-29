import argparse
import os
import cv2
import numpy as np
import api


def main(args):
    nb_images = len(os.listdir(args.ground_truth))  # does not take into account empty boxes

    # Initialize arrays to store metrics for each image
    precisions = np.zeros(nb_images)
    recalls = np.zeros(nb_images)
    f1_scores = np.zeros(nb_images)
    map_customs = np.zeros(nb_images)
    map50s = np.zeros(nb_images)
    map50_95s = np.zeros(nb_images)

    # Get all image files that have matching label files
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(args.images)
        if f.lower().endswith(image_extensions) and
           os.path.exists(os.path.join(args.ground_truth, os.path.splitext(f)[0] + ".txt"))
    ]

    # Loop through each valid image
    for idx, i in enumerate(image_files):

        # Read the image
        image = cv2.imread(os.path.join(args.images, i))

        # Load and process the prediction file (empty 'pred_list' if missing)
        pred_file = os.path.join(args.predictions, i[:-4] + ".txt")
        pred_list = api.txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
        # If predictions are in 4-element format (xywh), add dummy confidence of 1
        if len(pred_list) > 0 and len(pred_list[0]) == 4:
            pred_list = [x + (1,) for x in pred_list]
        # Convert predictions from YOLO format to (x_min, y_min, x_max, y_max, conf)
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]
        # Filter predictions by confidence threshold
        pred_list = [x for x in pred_list if x[4] > args.min_conf]
        # Remove overlapping predictions using IoU threshold
        pred_list = api.remove_overlapping_regions_wrt_iou(pred_list, overlap_treshold=args.max_overlap)

        # Load and process ground truth labels
        gt_file = os.path.join(args.ground_truth, i[:-4] + ".txt")
        gt_list = api.txt_to_tuple_list(gt_file)
        # Add dummy confidence for consistency and convert format
        gt_list = [x + (1,) for x in gt_list]
        gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]

        # Evaluate the predictions against ground truth using precision, recall, f1, etc.
        fp_list, metrics = api.evaluate_detections(pred_list, gt_list, 0.25)
        tp_list = [x for x in pred_list if x not in fp_list]

        # Optionally print verbose metrics for each image
        if args.verbose:
            print("precision :", metrics["precision"])
            print("recall :", metrics["recall"])
            print("f1_score :", metrics["f1_score"])
            # print("map@custom :", api.compute_map_50(pred_list, gt_list, label_bias=True))
            print("map@50 :", api.compute_map_50(pred_list, gt_list))
            print("map@50_95 :", api.compute_map_50_95(pred_list, gt_list))
            print()

        # Store the evaluation metrics for this image
        precisions[idx] = metrics["precision"]
        recalls[idx] = metrics["recall"]
        f1_scores[idx] = metrics["f1_score"]
        map_customs[idx] = api.compute_map_50(pred_list, gt_list, label_bias=True)
        map50s[idx] = api.compute_map_50(pred_list, gt_list)
        map50_95s[idx] = api.compute_map_50_95(pred_list, gt_list)

    # Compute and print average metrics across all evaluated images
    print("Average performance metrics over all images :")
    print("precision :", np.mean(precisions))
    print("recall :", np.mean(recalls))
    print("f1_score :", np.mean(f1_scores))
    # print("map@custom :", np.mean(map_customs))
    print("map@50 :", np.mean(map50s))
    print("map@50_95 :", np.mean(map50_95s))


def parse_args():
    parser = argparse.ArgumentParser(description="python inference_pipeline.py --input_folder my_image_folder")

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the images folder"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions folder"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the labels folder"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0,
        help="Minimum confidence threshold for predictions to be taken into account (default: 0)"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=1,
        help="Maximum overlap between detections (default: 1, which means no overlap threshold)"
    )
    parser.add_argument(
        "--no_map",
        action="store_true",
        help="Set this flag when map50 and map50-95 are irrelevant metrics"
    )
    # Parse arguments and run the main function
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)