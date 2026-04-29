import os
import cv2
import random
import sys
from sklearn.metrics import pairwise_distances
import numpy as np
import api
import inference_pipeline
import training_api


def random_sample(dataset, _, __, sample_size, seed):
    """
    Randomly samples a specified number of indices from a dataset of images.

    Args:
        dataset (str): Path to the dataset directory containing an "images" subfolder.
        _ : Unused placeholder argument (included for compatibility with a broader interface).
        __ : Unused placeholder argument (included for compatibility with a broader interface).
        sample_size (int): Number of images to randomly select.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        List[str]: A list of randomly selected image file paths from the dataset.
    """
    images_dir = os.path.join(dataset, "images")
    images = api.get_images(images_dir)

    return random_sample_from_image_names(images, sample_size, seed)


def random_sample_from_image_names(images, sample_size, seed):
    """
    Selects a random subset of image indices from a list of image paths.

    Args:
        images (List[str]): List of image file paths or names.
        sample_size (int): Number of samples to select randomly.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        np.ndarray: Array of randomly selected indices corresponding to the input image list.
    """
    np.random.seed(seed)

    random_indices = np.random.choice(np.arange(len(images)), size=sample_size, replace=False)
    return random_indices


def uniform_partition(dataset, _, __, sample_size, seed):
    """
    Selects a uniformly spaced subset of image indices from the dataset, with a random offset for variation.

    Args:
        dataset (str): Path to the dataset directory containing an 'images' subdirectory.
        _ (Any): Unused argument placeholder for interface consistency.
        __ (Any): Unused argument placeholder for interface consistency.
        sample_size (int): Number of uniformly spaced samples to select.
        seed (int): Seed for random offset to ensure reproducibility.

    Returns:
        List[int]: List of indices corresponding to uniformly selected images from the dataset.
    """
    images_dir = os.path.join(dataset, "images")
    images = api.get_images(images_dir)

    random.seed(seed)
    offset = random.randrange(len(images))
    step = len(images) / sample_size
    indices = [(round(i) + offset) % len(images) for i in np.arange(0, len(images), step)]
    return indices


def get_lowest_confidence_files(output_dir, n, aggregate):
    """
    Identifies the `n` files in a directory whose detections have the lowest average (or aggregated) confidence scores.

    Args:
        output_dir (str): Path to the directory containing YOLO-format prediction files, along with confidence scores
        n (int): Number of files with the lowest confidence to retrieve.
        aggregate (Callable): Function used to aggregate the list of confidence scores per file (e.g., mean, median).

    Returns:
        List[str]: Filenames (not full paths) of the `n` files with the lowest aggregated confidence scores.
    """
    confidence_data = []

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, 'r') as f:
            confidences = []
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:  # YOLO + confidence should be at least 6 elements
                    continue
                try:
                    confidence = float(parts[-1])
                    confidences.append(confidence)
                except ValueError:
                    continue

        if confidences:
            mean_conf = aggregate(confidences)
            confidence_data.append((filename, mean_conf))

    # Sort by mean confidence and return the n lowest
    confidence_data.sort(key=lambda x: x[1])
    return [filename for filename, _ in confidence_data[:n]]


def max_uncertainty(dataset, model, sample_size, aggregate):
    """
    Selects image indices corresponding to the `sample_size` most uncertain predictions
    based on the lowest aggregated confidence scores from model inference.

    Args:
        dataset (str): Path to the dataset directory (should contain an 'images' subdirectory).
        model (str): Path to the model file to use for inference.
        sample_size (int): Number of uncertain images to select.
        aggregate (Callable): Aggregation function (e.g., mean, median) to compute confidence per image.

    Returns:
        List[int]: Indices of images in the dataset with the lowest aggregated confidence scores.
    """
    images_dir = os.path.join(dataset, "images")
    imgs = api.get_images(images_dir)
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    lowest_confidence_indices = []
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_extensions.extend([ext.upper() for ext in image_extensions])
    for i in get_lowest_confidence_files("output", sample_size, aggregate):
        for ext in image_extensions:
            img = i[:-4] + ext
            if img in imgs:
                lowest_confidence_indices.append(imgs.index(img))
                break

    return lowest_confidence_indices


def max_mean_uncertainty_sample(dataset, _, model, sample_size, __):
    """
    Selects a sample of images from the dataset with the highest uncertainty based on
    the lowest mean confidence scores from model predictions.

    Args:
        dataset (str): Path to the dataset directory (must contain an 'images' subdirectory).
        _ (Any): Unused argument (placeholder for compatibility).
        model (str): Path to the model used for generating predictions.
        sample_size (int): Number of uncertain images to sample.
        __ (Any): Unused argument (placeholder for compatibility).

    Returns:
        List[int]: Indices of the `sample_size` most uncertain images, based on mean confidence.
    """
    return max_uncertainty(dataset, model, sample_size, np.mean)


def filter_empty_boxes(dataset, dir_to_filter):
    """
    Removes prediction files from a directory if their corresponding ground truth label
    file does not exist, effectively filtering out images with no annotations.

    Args:
        dataset (str): Path to the dataset directory containing a 'labels' subdirectory.
        dir_to_filter (str): Path to the directory containing prediction files to be filtered.

    Returns:
        None
    """
    labels_dir = os.path.join(dataset, "labels")
    images_dir = os.path.join(dataset, "images")

    _, labels = api.get_images_and_labels(images_dir, labels_dir)
    for f in os.listdir(dir_to_filter):
        if f not in labels:
            os.remove(os.path.join(dir_to_filter, f))


def diverse_sample(dataset, original_dataset, model, sample_size, seed):
    """
    Selects a diverse subset of images from a new dataset by maximizing feature-space
    average-detection diversity relative to a representative sample from the original dataset.

    Args:
        dataset (str): Path to the new dataset directory containing an 'images' subdirectory.
        original_dataset (str): Path to the original dataset used to compute diversity reference.
        model (str): Path to the trained detection model used for inference on the new dataset.
        sample_size (int): Number of diverse samples to select.
        seed (int): Random seed for reproducibility.

    Returns:
        List[int]: Indices of the selected images from the new dataset that maximize diversity.
    """
    # First select representative sample from the training set.
    # Diversity will then be computed wrt to that selection.
    original_images_dir = os.path.join(original_dataset, "images")
    original_labels_dir = os.path.join(original_dataset, "labels")
    original_images, original_labels = api.get_images_and_labels(original_images_dir, original_labels_dir)
    original_images.sort()
    original_labels.sort()
    representative_indices = training_api.make_representative_split(original_images_dir,
                                                                    original_labels_dir,
                                                                    min(sample_size, len(original_images)), seed)
    avg_detections = [training_api.make_average_detection(os.path.join(original_images_dir, original_images[i]),
                                             os.path.join(original_labels_dir, original_labels[i]), seed=seed)
                      for i in representative_indices]
    avg_detections = np.asarray(avg_detections)
    original_representative_features = training_api.extract_features(avg_detections)

    # Infer on the new images, to get bounding boxes
    images_dir = os.path.join(dataset, "images")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    # Calculate diversity and make selection
    labels_dir = "output"
    all_images = api.get_images(images_dir)
    valid_images, _ = api.get_images_and_labels(images_dir, labels_dir)
    valid_indices = training_api.make_representative_split(images_dir, labels_dir, sample_size,
                                                  seed=seed, original_features=original_representative_features)
    return api.convert_valid_to_real_indices(valid_indices, valid_images, all_images)


def supervised_sample(dataset, original_dataset, model, sample_size, seed, max_to_consider=500, supervised_ratio=0.75):
    """
    Selects a mixed sample from a new dataset using a supervised strategy:
    combines 'hard' samples from the original validation set (low mAP) and diverse samples from the new dataset.

    Args:
        dataset (str): Path to the new dataset directory.
        original_dataset (str): Path to the original dataset directory with validation data.
        model (str): Path to the detection model used for inference.
        sample_size (int): Total number of samples to select.
        seed (int): Random seed for reproducibility.
        max_to_consider (int, optional): Maximum number of new dataset images to consider for diversity sampling. Default is 500.
        supervised_ratio (float, optional): Ratio of supervised (hard) samples in the final sample. Default is 0.75.

    Returns:
        List[int]: Indices of selected images from the new dataset combining supervised hard samples and diverse samples.
    """
    random.seed(seed)

    # Run inference on original validation images
    original_val_images_dir = os.path.join(original_dataset, "val", "images")
    original_val_labels_dir = os.path.join(original_dataset, "val", "labels")
    sys.argv = f"inference_pipeline.py --input_folder {original_val_images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    original_val_images, original_val_labels = api.get_images_and_labels(original_val_images_dir,
                                                                         original_val_labels_dir)
    original_val_images.sort()
    original_val_labels.sort()
    original_val_size = len(original_val_images)

    # Compute mean average precision (mAP) with label bias for each validation image
    maps = np.zeros(original_val_size)
    for idx, i in enumerate(original_val_images):

        image = cv2.imread(os.path.join(original_val_images_dir, i))

        pred_file = os.path.join("output", i[:-4] + ".txt")
        pred_list = api.txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]

        gt_file = os.path.join(original_val_labels_dir, i[:-4] + ".txt")
        gt_list = api.txt_to_tuple_list(gt_file)
        gt_list = [x + (1,) for x in gt_list]
        gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]

        maps[idx] = api.compute_map_50(pred_list, gt_list, label_bias=True)

    # Determine number of worst-performing images to consider as supervised "hard" examples
    nb_worst = max(original_val_size // 10, 2)
    nb_supervised = round(sample_size * supervised_ratio)
    nb_diverse = sample_size - nb_supervised
    worst_indices = np.argsort(maps)[:nb_worst]
    worst_avg_detections = [training_api.make_average_detection(os.path.join(original_val_images_dir, original_val_images[i]),
                                             os.path.join(original_val_labels_dir, original_val_labels[i]), seed=seed)
                      for i in worst_indices]
    worst_avg_detections = np.asarray(worst_avg_detections)
    worst_features = training_api.extract_features(worst_avg_detections)

    # Run inference on the new dataset images
    new_images_dir = os.path.join(dataset, "images")
    sys.argv = f"inference_pipeline.py --input_folder {new_images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    # Limit number of new images considered for diversity to max_to_consider or sample_size (whichever is larger)
    all_new_images = api.get_images(new_images_dir)
    new_images, new_labels = api.get_images_and_labels(new_images_dir, "output")
    max_to_consider = max(sample_size, max_to_consider)
    if len(new_images) > max_to_consider:
        # Uniformly select max_to_consider images to consider for supervised sampling
        indices_to_consider = uniform_partition(dataset, None, None, max_to_consider, seed)
        new_images = [new_images[i] for i in indices_to_consider]
        new_labels = [new_labels[i] for i in indices_to_consider]
    # Compute average detections for the filtered new images
    new_avg_detections = [training_api.make_average_detection(os.path.join(new_images_dir, new_images[i]),
                                                          os.path.join("output", new_labels[i]),
                                                          seed=seed)
                      for i in range(len(new_images))]
    new_avg_detections = np.asarray(new_avg_detections)
    new_features = training_api.extract_features(new_avg_detections)

    # Select new images closest to hard examples (supervised samples)
    dists = pairwise_distances(new_features, worst_features)
    min_dists = dists.min(axis=1)
    supervised_indices = list(np.argsort(min_dists)[:nb_supervised])
    supervised_indices = api.convert_valid_to_real_indices(supervised_indices, new_images, all_new_images)
    indices = list(supervised_indices)

    # Select diverse samples for remaining quota
    diverse_indices = diverse_sample(dataset, original_dataset, model, nb_diverse, seed)

    # Supervised and diverse selections might have overlapped.
    # Randomly sample the number of images needed to meet sample size
    n_random = 0
    for idx in diverse_indices:
        if idx not in supervised_indices:
            indices.append(idx)
        else:
            n_random += 1

    if len(new_images) - len(indices) < n_random:
        raise ValueError("Demanded sample size is greater than the dataset to sample from"
                         "(with empty boxes discarded). Please ask for a smaller sample size")
    remaining_indices = [all_new_images.index(new_images[i]) for i in range(len(new_images))
                         if all_new_images.index(new_images[i]) not in indices]
    random.shuffle(remaining_indices)
    for i in range(n_random):
        indices.append(remaining_indices[i])

    return indices


def test_sample(sampling_method, model, sample_name="test", k=30):
    """
    Tests the given sampling method, producing a selection in output directory 'sample_name'

    Args:
        sampling_method (Callable): Python sampling method which takes 5 parameters (dataset, original_dataset,
                                    model, sample_size, seed).
        model (str): Path to the detection model used for inference.
        sample_name (str, optional): Path to the output directory used to store selection. Defaults to 'test'.
        k (int, optional): Desired selection size. Defaults to 30.

    Returns:
        None
    """
    print(f"Preparing selection {sample_name}...")
    original_dataset = "whole_dataset"
    # all_images = api.get_images(os.path.join("whole_dataset", "images"))
    # dataset0_indices = [all_images.index(f) for f in os.listdir(os.path.join("dataset0", "images"))]
    # remaining_indices = [i for i in range(len(all_images)) if i not in dataset0_indices]
    # remaining_dataset = "remaining"
    # api.make_set_from_indices(remaining_dataset, os.path.join("whole_dataset", "images"),
    #                           os.path.join("whole_dataset", "labels"), remaining_indices)
    remaining_dataset = "Test_Set"

    labels_dir = os.path.join(remaining_dataset, "labels")
    images_dir = os.path.join(remaining_dataset, "images")

    indices = sampling_method(remaining_dataset, original_dataset, model, k, 69)
    api.make_set_from_indices(sample_name, images_dir, labels_dir, indices)

# test_sample(random_sample, None, "random_selection")
# test_sample(uniform_partition, None, "uniform_partition")
#test_sample(max_mean_uncertainty_sample, "model/final_23.pt", "max_uncertainty", k=3)
#test_sample(diverse_sample, "dataset0.pt", "diverse", k=4)
#test_sample(supervised_sample, "model/final_23.pt", "super", k=4)
