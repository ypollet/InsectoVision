import argparse
import os
import shutil
import sys
import api
import training_pipeline

seed = 69


def main(args):

    # CONCURRENT REHEARSAL CONTINUAL LEARNING (Not Sequential)

    # Warn user if "new_dataset" directory already exists, unless silent mode is active
    api.warn_user_if_directory_exists("new_dataset", silent=args.silent)
    # Merge the original dataset with new images, using validation replacement strategy.
    # If --max_old_samples is set, cap the historical training data carried forward to a
    # diversity-sampled subset instead of replaying the full history every round (bounds
    # training set growth across repeated retrain.py calls while still rehearsing old
    # appearance diversity to avoid catastrophic forgetting on previously seen insect types).
    if args.max_old_samples is not None and args.max_old_samples > 0:
        S, k = api.merge_datasets_diverse(args.dataset, args.new_images, "new_dataset", seed=seed,
                                          max_old_train_samples=args.max_old_samples)
    else:
        S, k = api.merge_datasets(args.dataset, args.new_images, "new_dataset", seed=seed)
    # Calculate scaling factor for training based on the proportion of new data (ensuring min factor of 0.01)
    factor = max(k / (S + k), 0.01)

    # Get number of training and validation images in the merged dataset
    new_train_size = len(os.listdir(os.path.join("new_dataset", "train", "images")))
    new_val_size = len(os.listdir(os.path.join("new_dataset", "val", "images")))
    # Print dataset merge summary if verbose mode is on
    if args.verbose:
        print("Original dataset size:", S, "\nNumber of images to add:", k,
              "\nNew dataset size:", new_train_size + new_val_size,
              "\nNew validation ratio", new_val_size / (new_train_size + new_val_size))

    original_argv = list(sys.argv)
    # Compute new epoch and step counts, scaled by the dataset factor, with lower bounds
    new_epochs = max(int(args.original_epochs * factor), 3)
    new_steps = max(int(args.original_nb_steps * factor), 3)
    # Construct arguments to launch fine-tuning training on merged dataset
    training_args = f"training_pipeline.py --dataset new_dataset --fine_tuning_steps {new_steps} " \
                     f"--model {args.model} --lr0 {args.original_lr0 * factor} --gpu {args.gpu} " \
                    f"--epochs {new_epochs} --batch_init {args.original_batch} --batch_min {args.original_batch} --img_size {args.img_size} " \
                    f"--replace_all --patience {max(new_epochs // 3, 2)} --no_split".split()
    # Add optional flags based on user input
    if args.detection_only:
        training_args.append("--detection_only")
    if args.classification_only:
        training_args.append("--classification_only")
    if args.verbose:
        training_args.append("--verbose")
    # Override sys.argv so the training pipeline receives the constructed arguments
    sys.argv = list(training_args)
    # Parse the new arguments and run the training
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)

    if not args.classification_only:
        # Backup the trained detection model and remove temporary output
        api.warn_user_if_file_exists("new_model.pt")
        shutil.copy2("output.pt", "new_model.pt")
        os.remove("output.pt")

    if not args.detection_only:
        # Backup the trained classification model and remove temporary output
        api.warn_user_if_file_exists("new_model.keras")
        shutil.copy2("output.keras", "new_model.keras")
        os.remove("output.keras")

    # If doing full pipeline (not detection-only or classification-only) and high precision is requested
    if (not (args.detection_only or args.classification_only)) and args.high_precision:

        # Construct arguments for high-precision fine-tuning using the new model and classifier
        training_args = f"training_pipeline.py --dataset new_dataset --fine_tuning_steps {new_steps} " \
                        f"--model new_model.pt --heatmap_extractor new_model.keras --lr0 {args.original_lr0 * factor} " \
                        f"--epochs {new_epochs} --batch_init {args.original_batch} --img_size {args.img_size} " \
                        f"--replace_all --patience {max(new_epochs // 3, 2)} --no_split".split()
        if args.verbose:
            training_args.append("--verbose")
        # Override sys.argv again and rerun training for high-precision refinement
        sys.argv = list(training_args)
        training_args = training_pipeline.parse_args()
        training_pipeline.main(training_args)

        # Backup high-precision model and clean up
        api.warn_user_if_file_exists("new_high_detection_model.pt")
        shutil.copy2("output.pt", "new_high_detection_model.pt")
        os.remove("output.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="python retrain.py --dataset my_dataset --new_images my_new_images "
                                                 "--model my_original_model.pt")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset, in standard yolo format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--new_images",
        type=str,
        required=True,
        help="Path to the new images to add to the dataset, in standard yolo "
             "format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained detection model to fine-tune on new samples"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="Gpu to use, the default is the macos standard (default: cuda:0)"
    )
    parser.add_argument(
        "--original_lr0",
        type=float,
        default=0.01,
        help="Initial learning rate which was used to train the input model (default: 0.01)"
    )
    parser.add_argument(
        "--original_batch",
        type=int,
        default=16,
        help="Initial batch size which was used to train the input model (default: 16)"
    )
    parser.add_argument(
        "--original_epochs",
        type=int,
        default=20,
        help="Initial number of epochs which was used to train the input model (default: 20)"
    )
    parser.add_argument(
        "--original_nb_steps",
        type=int,
        default=23,
        help="Initial number of fine-tuning steps which was used to train the input model (default: 10)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Replace all existing directories without warning user"
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Disables corrector training, only trains detector"
    )
    parser.add_argument(
        "--classification_only",
        action="store_true",
        help="Disables detector training, only trains corrector"
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="Enables high-precision low-recall model training"
    )
    parser.add_argument(
        "--max_old_samples",
        type=float,
        default=0.75,
        help="If set to a positive value, bounds the amount of historical training data carried "
             "into the merged dataset, selected via feature-based diversity sampling rather than "
             "kept in full. A value strictly between 0 and 1 is treated as the target share of old "
             "data in the merged training set (e.g. 0.75 aims for 75%% old / 25%% new); if the "
             "archive doesn't hold enough old images to reach that share, all of it is kept instead "
             "(sets are just merged). A value >= 1 is treated as an absolute image count. Set to 0 "
             "or a negative value to disable and keep all historical data (default: 20)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)