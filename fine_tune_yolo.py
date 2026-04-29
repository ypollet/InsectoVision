import argparse
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

# Directories for training monitoring
run_dir = os.path.join("runs", "detect")
abs_run_path = os.path.join(os.getcwd(), run_dir)
attempt_dir = "attempts"

# Hyper-parameters for re-runs
WAIT_ON_FAIL = 0 # waiting time for cooling purposes
INCREMENT_WAIT = 0 # waiting increment on subsequent fails
DECREMENT_BATCH = 1 # batch decrement on subsequent fails
MAX_TRY = 4 # max number of re-runs


def run_ok(run_folder):
    """
    Checks if a run folder contains a confusion matrix output image. YOLO only generates this matrix
    when training has completed, so it can be used to detect failure.

    Args:
        run_folder (str): Name of the folder corresponding to a specific run.

    Returns:
        bool: True if 'confusion_matrix.png' exists in the run folder, False otherwise.
    """
    return "confusion_matrix.png" in os.listdir(os.path.join(abs_run_path, run_folder))


def last_run_ok():
    """
    Checks whether the most recent training run completed successfully.

    Returns:
        bool: True if the latest run did not crash (i.e., completed and produced expected output),
              False otherwise.
    """
    train_folder = os.listdir(abs_run_path)
    train_folder.sort(key=lambda f: int(f[5:]) if len(f) > 5 else 1)
    return run_ok(train_folder[-1])


def delete_last_run(keep_copy=False):
    """
    Deletes the most recent training run directory, with an option to back it up in the `attempt_dir`
    directory.

    Args:
        keep_copy (bool, optional): If True, a copy of the run is saved to `attempt_dir` before deletion.
                                    Defaults to False.

    Returns:
        str: Name of the deleted run folder.
    """
    train_folder = os.listdir(abs_run_path)
    train_folder.sort(key=lambda f: int(f[5:]) if len(f) > 5 else 1)
    if keep_copy:
        copy_id = len(os.listdir(attempt_dir))
        shutil.copytree(os.path.join(abs_run_path, train_folder[-1]),
                        os.path.join(attempt_dir, train_folder[-1] + f"_{copy_id}"))
    shutil.rmtree(os.path.join(abs_run_path, train_folder[-1]))
    return train_folder[-1]


def get_last_valid_run():
    """
    Finds the most recent training run that completed successfully.

    Returns:
        int: The run ID of the latest successful run.
    """
    train_folder = os.listdir(abs_run_path)
    train_folder.sort(key=lambda f: int(f[5:]) if len(f) > 5 else 1, reverse=True)
    run_id = int(train_folder[0][5:]) if len(train_folder[0]) > 5 else 1
    for name in train_folder:
        if run_ok(name):
            return run_id
        else:
            run_id -= 1


def get_best_model(run_dir=abs_run_path):
    """
    Finds the model checkpoint with the highest mAP@0.5 score across all training runs.

    Args:
        run_dir (str, optional): Path to the directory containing training run subfolders.
                                 Defaults to `abs_run_path`.

    Returns:
        tuple: (best_model_path, best_run_dir, best_map50) where:
            - best_model_path (str): Path to the best model checkpoint (`best.pt`).
            - best_run_dir (str): Name of the run directory containing the best model.
            - best_map50 (float): The highest mAP@0.5 score found.
    """
    best_map50 = 0
    best_model = None
    best_dir = None
    for train_dir in os.listdir(run_dir):
        map50 = get_map50(os.path.join(run_dir, train_dir))
        if best_map50 < map50:
            best_map50 = map50
            best_model = os.path.join(run_dir, train_dir, "weights", "best.pt")
            best_dir = train_dir
    print(f"Best model found in {run_dir}, achieved map50 {best_map50}")
    return best_model, best_dir, best_map50


def get_attempt_id(attempt_name):
    """
    Extracts the numeric attempt ID from an attempt folder name.

    Args:
        attempt_name (str): Name of the attempt folder, expected to end with an underscore and a number
                            (e.g., 'current_train_3').

    Returns:
        int: The extracted attempt ID.

    Raises:
        ValueError: If the folder name doesn't match the expected format.
    """
    offset = -1
    for i in range(len(attempt_name) - 1, -1, -1):
        if attempt_name[i] == "_":
            offset = i + 1
    if offset == -1:
        raise ValueError("Attempt folder name doesn't have the right format, "
                         "which is expected to end with an underscore and a number "
                         " (e.g., 'current_train_3')")
    return int(attempt_name[offset:])


def get_map50(train_path):
    """
    Retrieves the highest mAP@0.5 (Box) score from a training run's results.

    Args:
        train_path (str): Path to the training run directory containing 'results.csv'.

    Returns:
        float: The highest recorded mAP@0.5 score. Returns 0 if the results file is missing.
    """
    results_path = os.path.join(train_path, "results.csv")
    if not os.path.exists(results_path):
        return 0
    df = pd.read_csv(results_path)
    best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
    map50 = best_row['metrics/mAP50(B)']
    return map50


def train_yolo(freeze_layers, batch_size, lr0, model_path, dataset, gpu, patience, epochs, img_size, batch_size_list, try_id=1):
    """
    Trains a YOLO model with specified parameters and implements retry logic for failed runs.

    Args:
        freeze_layers (int): Number of model layers to freeze during training.
        batch_size (int): Initial batch size to use.
        lr0 (float): Initial learning rate.
        model_path (str): Path to the pre-trained YOLO model.
        dataset (str): Directory containing the YOLO dataset (with data.yaml).
        gpu (int or str): GPU identifier for training.
        patience (int): Early stopping patience (YOLO setting).
        epochs (int): Number of training epochs.
        img_size (int): Training image size.
        batch_size_list (list[int]): List to track the batch sizes used for each training attempt.
        try_id (int, optional): Identifier for the current training attempt. Defaults to 1.

    Returns:
        None
    """
    print(f"\n\nStarting attempt number {try_id}...")

    # Remove the attempts directory before starting a new attempt (only on first attempt)
    if try_id == 1:
        if os.path.exists(attempt_dir):
            shutil.rmtree(attempt_dir)

    # Prepare training command for YOLOv8
    path_to_data = os.path.join(dataset, "data.yaml")
    command = (
        f"yolo task=detect mode=train model={model_path} "
        f"data={path_to_data} project={abs_run_path} epochs={epochs} imgsz={img_size} "
        f"plots=True device={gpu} batch={batch_size} optimizer=Adam patience={patience} "
        f"freeze={freeze_layers} lr0={lr0} lrf={0.01}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    # If the run failed, manage recovery logic
    if not last_run_ok():
        if not os.path.exists(attempt_dir):
            os.makedirs(attempt_dir)

        max_attempt = MAX_TRY
        if try_id >= max_attempt:
            # Maximum attempts reached, fallback to best past attempt
            print(f"{max_attempt} runs failed in a row, skipping this pipelining step")
            train_folder_name = delete_last_run(keep_copy=True)
            print("Storing best attempt...\n\n")
            _, best_train_dir, _ = get_best_model(run_dir=attempt_dir)
            if best_train_dir is None:
                raise ValueError(f"Pipelining step never completed any epoch")

            # Recovering best attempt's ID and adjusting batch size accordingly
            best_attempt_id = get_attempt_id(best_train_dir)
            batch_size_list.append(batch_size + ((try_id - 1) * DECREMENT_BATCH) - (best_attempt_id * DECREMENT_BATCH))
            shutil.copytree(os.path.join(attempt_dir, best_train_dir),
                            os.path.join(abs_run_path, train_folder_name))
            shutil.rmtree(attempt_dir)
        else:
            # Wait and retry training with smaller batch size
            waiting_time = WAIT_ON_FAIL + (try_id - 1) * INCREMENT_WAIT
            print(
                f"\n\nLast fine-tuning step has failed, waiting {waiting_time/60} minutes to cool off computer...")
            for i in range(waiting_time):
                time.sleep(1)
                print(".", end="\n" if (i + 1) % 60 == 0 else "", flush=True)
            new_batch_size = max(batch_size - DECREMENT_BATCH, 8)
            if batch_size != new_batch_size:
                print(f"Batch size will be reduced from {batch_size} to {new_batch_size}")
            delete_last_run(keep_copy=True)

            # Recursive retry with updated batch size
            train_yolo(freeze_layers, new_batch_size, lr0, model_path,
                       dataset, gpu, patience, epochs, img_size, batch_size_list, try_id=try_id + 1)

    # If run succeeded and attempt_dir exists, compare result with previous attempts
    elif os.path.exists(attempt_dir):
        print(f"Attempt completed, comparing with previous attempts...")
        run_id = get_last_valid_run()
        len_run_dir = 5 if run_id == 1 else 5 + len(str(run_id))
        run_path = os.path.join(abs_run_path, f"train{run_id}" if run_id > 1 else "train")
        map_50_completed = get_map50(run_path)
        _, best_attempt_dir, map_50_attempts = get_best_model(run_dir=attempt_dir)

        # Replace run with better-performing attempt if necessary
        if map_50_attempts > map_50_completed:
            print(f"Previous attempt scored best map50 {map_50_attempts}")
            shutil.rmtree(run_path)
            shutil.copytree(os.path.join(attempt_dir, best_attempt_dir), run_path)
            attempt_id = int(best_attempt_dir[len_run_dir + 1:])
            batch_size_list.append(batch_size + ((try_id - 1) * DECREMENT_BATCH) - (attempt_id * DECREMENT_BATCH))
        else:
            print(f"This attempt scored best map50 {map_50_completed}")
            batch_size_list.append(batch_size)
        shutil.rmtree(attempt_dir)

    # Run succeeded at first try, no comparison needed
    else:
        batch_size_list.append(batch_size)


def log_best_map(run_id, results, batch_size, step, run_directory=run_dir):
    """
    Logs the best mAP@50 score of a given run along with its configuration.

    Args:
        run_id (int): Identifier of the training run.
        results (list[tuple]): A list to append the results tuple to.
        batch_size (int): The batch size used for the run.
        step (int): Step used to compute the varying number of frozen layers.
        run_directory (str, optional): Path to the directory containing training runs. Defaults to run_dir.

    Returns:
        None
    """
    train_path = os.path.join(run_directory, "train" if run_id == 1 else f"train{run_id}")
    best_map50 = get_map50(train_path)
    tup_to_append = (run_id, 22 - (run_id - 1)*step, batch_size, best_map50)
    results.append(tup_to_append)


def main(args):
    # Delete run and attempt directories if they exist
    if os.path.exists(attempt_dir):
        shutil.rmtree(attempt_dir)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # Initial settings
    freeze_layers = 22
    batch_size = args.batch_init
    lr0 = args.lr0
    results = []
    batch_size_list = []

    # First run with yolov8s.pt
    train_yolo(freeze_layers, batch_size, lr0, args.model, args.dataset, args.gpu, args.patience, args.epochs, args.img_size, batch_size_list)

    # Define parameters needed for gradual layer unfreezing and learning rate decreasing
    step = 22 / (args.fine_tuning_steps - 1)
    freeze_limit = (22 - step * (args.fine_tuning_steps - 1)) - 1
    lr_decrease_factor = (args.lr_final / lr0) ** (1 / args.fine_tuning_steps)

    # Run subsequent training iterations from (22 - step) frozen layers down to 0
    for run_id, freeze_layers in enumerate(np.arange(22 - step, freeze_limit, -step), start=2):
        # Gradually decrease learning rate
        lr0 *= lr_decrease_factor

        # Decrease batch size, but not below 8
        batch_size = max(args.batch_min, batch_size - args.batch_decrease_rate)

        # Get best previously trained model for weight initialization
        model_path, _, _ = get_best_model()
        if model_path is None:
            model_path = args.model

        # Train and monitor with given parameters
        train_yolo(round(freeze_layers), round(batch_size), lr0, model_path, args.dataset,
                   args.gpu, args.patience, args.epochs, args.img_size, batch_size_list)

    # Log run parameters and results to the 'results' list
    for rid in range(1, args.fine_tuning_steps + 1):
        log_best_map(rid, results, batch_size_list[rid - 1], step)

    # Print final results table
    print("\nFinal Training Results:")
    print("{:<5} {:<10} {:<8} {:<12}".format("Run", "Freeze", "Batch", "mAP50"))
    print("-" * 50)
    for run_id, freeze, batch, map50 in results:
        print(f"{run_id:<5} {round(freeze):<10} {batch:<8} {map50:<12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python fine_tune_yolo.py --dataset my_dataset")

    # Add command-line arguments
    parser.add_argument(
        "--fine_tuning_steps",
        type=int,
        default=5,
        help="Number of fine-tuning runs, with a constant rate of layer "
             "unfreezing down to 0 frozen layers (default: 5)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="mps",
        help="Gpu to use, the default is the macos standard (default: mps)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lr_final",
        type=float,
        default=-1,
        help="Final run learning rate (default: lr0/10)"
    )
    parser.add_argument(
        "--batch_init",
        type=int,
        default=16,
        help="Initial batch size (default: 16)"
    )
    parser.add_argument(
        "--batch_min",
        type=int,
        default=8,
        help="Minimal batch size from which it will start plateauing (default: 8)"
    )
    parser.add_argument(
        "--batch_decrease_rate",
        type=float,
        default=1,
        help="Rate at which batch size is reduced for each subsequent run (default: 1 per run)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximal number of epochs per run (default: 20)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience fo early stopping (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/yolov8s.pt",
        help="Pretrained detection model to fine-tune on dataset (default: yolov8s.pt)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset in yaml format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)