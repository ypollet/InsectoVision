import os
import shutil
import numpy as np
import api
import random
import sys
import training_pipeline
import retrain
import inference_pipeline
import performance
import subsample

dataset = "whole_dataset"
test_set = "Test_Set"
seed = 69
random.seed(seed)
np.random.seed(seed)
silent = True
iterations = 10
ft_steps = 10


def train(dataset, no_high_precision=True, no_split=True):
    """
    Executes the full training pipeline for a given dataset, including an optional high-precision refinement step.

    Args:
        dataset (str): path of the dataset to train on.
        no_high_precision (bool, optional): If True, skips the high-precision secondary training step. Defaults to True.
        no_split (bool, optional): If True, keeps the original validation split and
                                   avoids re-splitting the dataset. Defaults to True.

    Returns:
        None
    """
    sys.argv = f"training_pipeline.py --dataset {dataset} " \
               f"--verbose --replace_all --fine_tuning_steps {ft_steps}".split()
    if no_split:
        sys.argv.append("--no_split")
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)

    api.warn_user_if_file_exists(f"{dataset}.pt", silent=silent)
    api.warn_user_if_file_exists(f"{dataset}.keras", silent=silent)
    os.rename("output.pt", f"{dataset}.pt")
    os.rename("output.keras", f"{dataset}.keras")

    if not no_high_precision:
        sys.argv = f"training_pipeline.py --dataset {dataset} --heatmap_extractor {dataset}.keras " \
                   f"--verbose --replace_all --fine_tuning_steps {ft_steps}".split()
        training_args = training_pipeline.parse_args()
        training_pipeline.main(training_args)

        api.warn_user_if_file_exists(f"{dataset}_high_precision.pt", silent=silent)
        os.rename("output.pt", f"{dataset}_high_precision.pt")


def print_performance(images, labels, predictions, min_conf=0.0, no_map=False):
    """
    Runs the performance evaluation script on a set of predictions and prints the results.

    Args:
        images (str): Path to the directory containing image files.
        labels (str): Path to the ground-truth labels directory.
        predictions (str): Path to the predictions directory.
        min_conf (float, optional): Minimum confidence threshold for predictions. Defaults to 0.0.
        no_map (bool, optional): If True, disables mAP calculation. Defaults to False.

    Returns:
        None
    """
    sys.argv = f"performance.py --images {images} " \
               f"--ground_truth {labels} --predictions {predictions} --min_conf {min_conf}".split()
    if no_map:
        sys.argv.append("--no_map")
    args = performance.parse_args()
    performance.main(args)


def assess_performance(images_dir, labels_dir, model, classifier, high_precision_model=None, include_corrector=True):
    """
    Evaluates the performance of a YOLO detector (and optionally a post-detection corrector/classifier)
    on a given dataset by running inference and printing performance metrics.

    Args:
        images_dir (str): Path to the directory containing input images.
        labels_dir (str): Path to the directory containing ground truth labels.
        model (str): Path to the primary detection model.
        classifier (str): Path to the classifier model used for correction.
        high_precision_model (str, optional): Optional high-precision model for secondary evaluation. Defaults to None.
        include_corrector (bool, optional): Whether to include classifier correction in the evaluation. Defaults to True.

    Returns:
        None
    """
    print(f"Inferring with detector {model}...\n")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
               f"--silent --model {model} --detection_only --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    print(f"Performance of detector {model}:")
    print_performance(images_dir, labels_dir, "output")
    print()

    if include_corrector:

        print(f"Inferring with detector {model} and corrector {classifier}...\n")
        sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
                   f"--silent --model {model} --classifier {classifier} --write_conf".split()
        args = inference_pipeline.parse_args()
        inference_pipeline.main(args)

        print(f"Performance of detector {model} + corrector {classifier}:")
        print_performance(images_dir, labels_dir, "output", no_map=True)
        print()

    if high_precision_model is not None:
        print(f"Inferring with high_precision detector {high_precision_model}...\n")
        sys.argv = f"inference_pipeline.py --input_folder {images_dir} --high_precision " \
                   f"--silent --model {high_precision_model} --classifier {classifier} --write_conf".split()
        args = inference_pipeline.parse_args()
        inference_pipeline.main(args)

        print(f"Performance of high-precision detector {high_precision_model}:")
        print_performance(images_dir, labels_dir, "output")
        print()


def fine_tune(model, original_dataset, new_images, no_high_precision=True):
    """
    Fine-tunes a previously trained YOLO model using a new set of images and optionally produces a high-precision version.

    Args:
        model (str): Path to the pre-trained model to fine-tune.
        original_dataset (str): Name of the dataset used to originally train the model.
        new_images (str): Directory containing new images for fine-tuning.
        no_high_precision (bool, optional): If True, skips training a high-precision variant of the model. Defaults to True.

    Returns:
        None
    """
    print(f"\n\nFine-tuning model {model}, which was trained with dataset {original_dataset}, "
          f"on new images {new_images}...\n\n")
    sys.argv = f"retrain.py --dataset {original_dataset} --new_images {new_images} " \
               f"--verbose --model {model} --original_nb_steps {ft_steps}".split()
    if not no_high_precision:
        sys.argv.append("--high_precision")
    if silent:
        sys.argv.append("--silent")
    training_args = retrain.parse_args()
    retrain.main(training_args)

    api.warn_user_if_file_exists(f"{new_images}.pt", silent=silent)
    api.warn_user_if_file_exists(f"{new_images}.keras", silent=silent)
    os.rename("new_model.pt", f"{new_images}.pt")
    os.rename("new_classifier.keras", f"{new_images}.keras")
    if not no_high_precision:
        api.warn_user_if_file_exists(f"{new_images}_high_precision.pt", silent=silent)
        os.rename("new_high_precision_model.pt", f"{new_images}_high_precision.pt")
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)


def merge_and_train(_, original_dataset, new_images, no_high_precision=True):
    print(f"\n\nMerging {original_dataset} with {new_images} and training normally...\n\n")
    api.merge_datasets(original_dataset, new_images, "new_dataset", seed=seed)
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)
    train(new_images, no_high_precision=no_high_precision)


def simulate_active_learning(sampling_strategy=subsample.random_sample,
                             initial_sampling_strategy=subsample.uniform_partition, training_strategy=fine_tune,
                             high_precision=False, train_dataset0=True, log_directory=None):
    """
    Simulates an active learning pipeline by iteratively training and sampling data.

    Args:
        sampling_strategy (callable): Function to select new data points from the remaining unlabeled pool.
        initial_sampling_strategy (callable): Function to select the initial labeled dataset.
        training_strategy (callable): Function to train the model, e.g., `fine_tune`.
        high_precision (bool): Whether to include high-precision models in training.
        train_dataset0 (bool): If True, samples the initial dataset and trains
                               before starting active learning rounds.
        log_directory (str or None): Path to save training logs/runs. If None, logs are deleted.

    Returns:
        None
    """
    # Warn and prepare logging directory if needed
    if log_directory is not None:
        api.warn_user_if_directory_exists(log_directory, silent=silent)

    # Initial data setup
    labels_dir = os.path.join(dataset, "labels")
    images_dir = os.path.join(dataset, "images")

    images, labels = api.get_images_and_labels(images_dir, labels_dir)
    all_images = api.get_images(images_dir)

    # Select initial training set indices using the initial strategy
    indices = initial_sampling_strategy(dataset, None, None, len(all_images) // iterations, seed)
    if train_dataset0:
        # Samples initial dataset 'dataset0' from indices
        api.make_set_from_indices("dataset0", images_dir, labels_dir, indices, silent=silent)
    # Create remaining pool from images not selected initially
    api.make_set_from_indices("remaining", images_dir, labels_dir,
                              [i for i in range(len(all_images)) if i not in indices], silent=silent)

    # Initial model training
    if train_dataset0:
        train("dataset0", no_high_precision=not high_precision, no_split=False)
        api.warn_user_if_directory_exists("runs0", silent=silent, make_dir=False)
        shutil.move("runs", "runs0")

    # Active learning iterations
    for i in range(1, iterations):
        # Rename 'remaining' to 'current_remaining' for this round
        api.warn_user_if_directory_exists("current_remaining", silent=silent, make_dir=False)
        os.rename("remaining", "current_remaining")
        labels_dir = os.path.join(f"current_remaining", "labels")
        images_dir = os.path.join(f"current_remaining", "images")

        images, labels = api.get_images_and_labels(images_dir, labels_dir)

        # Select a new batch from the remaining pool
        indices = sampling_strategy("current_remaining", f"dataset{i - 1}",
                                    f"dataset{i - 1}.pt", len(images) // (iterations - i), seed)
        api.make_set_from_indices(f"dataset{i}", images_dir, labels_dir, indices, silent=silent)
        api.make_set_from_indices("remaining", images_dir, labels_dir,
                                  [i for i in range(len(images)) if i not in indices], silent=silent)

        # Re-train the model using the new dataset
        training_strategy(f"dataset{i - 1}.pt", f"dataset{i - 1}",
                          f"dataset{i}", no_high_precision=not high_precision)
        # Move training logs to the round's subdirectory if log parent-directory specified
        if log_directory is not None:
            api.warn_user_if_directory_exists(os.path.join(log_directory, f"runs{i}"), silent=silent, make_dir=False)
            shutil.move("runs", os.path.join(log_directory, f"runs{i}"))

        # Clean up temporary folder for this iteration
        shutil.rmtree("current_remaining")


images_dir = os.path.join(dataset, "images")
labels_dir = os.path.join(dataset, "labels")
test_images_dir = os.path.join(test_set, "images")
test_labels_dir = os.path.join(test_set, "labels")
strategies = [subsample.random_sample, subsample.max_mean_uncertainty_sample, subsample.diverse_sample, subsample.supervised_sample, subsample.uniform_partition]
strategy_names = ["random_sample", "uncertainty", "diversity", "supervised", "uniform"]

# os.makedirs("remaining")
# os.makedirs("remaining/images")
# os.makedirs("remaining/labels")
# images, labels = api.get_images_and_labels(images_dir, labels_dir)
# for i, image in enumerate(images):
#     if image not in os.listdir("dataset1/images"):
#         shutil.copy2(os.path.join(images_dir, image), os.path.join("remaining/images", image))
#         shutil.copy2(os.path.join(labels_dir, labels[i]), os.path.join("remaining/labels", labels[i]))
#
# exit()

# Repeat for every subsampling strategy
for strat_id, strat in list(enumerate(strategies)):
    # Launch AL simulation with given subsampling strategy,
    # sampling and training initial dataset0 only for the first strategy
    simulate_active_learning(sampling_strategy=strat, training_strategy=fine_tune, train_dataset0=(strat_id == 0),
                             log_directory=strategy_names[strat_id])

    # Move every output detectors and classifiers to their relevant log directories,
    # for subsequent performance assessment
    for i in range(iterations):
        prefix = f"dataset{i}"
        model = prefix + ".pt"
        classifier = prefix + ".keras"
        if i > 0:
            shutil.move(model, os.path.join(strategy_names[strat_id], model))
            shutil.move(classifier, os.path.join(strategy_names[strat_id], classifier))

# Assess performance of each AL round for every subsampling strategy
print(f"\n### Evaluating performance of initial model trained on dataset0 ###\n".upper())
assess_performance(test_images_dir, test_labels_dir, "dataset0.pt", "dataset0.keras", include_corrector=False)
for strat in strategy_names:
    print(f"\n### Evaluating performance of sampling strategy {strat} ###\n".upper())
    for i in range(1, iterations):
        prefix = f"dataset{i}"
        model = os.path.join(strat, prefix + ".pt")
        classifier = os.path.join(strat, prefix + ".keras")
        assess_performance(test_images_dir, test_labels_dir, model, classifier, include_corrector=False)
        print()
