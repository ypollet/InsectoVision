import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import tensorflow as tf
import os
import inference_pipeline
import api
import training_api

seed = 69

def main(args):
    # High-precision training does not need post-detection filtering
    if args.heatmap_extractor is not None:
        args.detection_only = True
    # Must be either detection or classification or both
    if args.detection_only and args.classification_only:
        raise ValueError("--detection_only and --classification_only cannot be specified at the same time,"
                         " as they are mutually exclusive")
    # Default lr_final is set to lr0 / 10
    if args.lr_final == -1:
        args.lr_final = args.lr0 / 10
    # Default batch_decrease_rate is set to a uniform step from batch_init down to batch_min
    if args.batch_decrease_rate == -1:
        args.batch_decrease_rate = (args.batch_init - args.batch_min) / (args.fine_tuning_steps - 1)

    # Unless no_split is True, split dataset in train and validation and build yaml file
    if not args.no_split:
        # Configuration
        dataset_dir = args.dataset
        original_images_dir = os.path.join(dataset_dir, "images")
        original_labels_dir = os.path.join(dataset_dir, "labels")
        output_yaml_path = os.path.join(dataset_dir, "data.yaml")
        class_names = ["insect"] # TODO update class names
        val_split = 0.2
        random.seed(seed)

        # Get all image files that have matching label files
        image_files, label_files = api.get_images_and_labels(original_images_dir, original_labels_dir)

        # Split into train and val
        train_files, val_files = train_test_split(image_files, test_size=val_split, random_state=seed)

        # For high-precision training, build an F-CNN from the heatmap_extractor classifier
        fcnn = None
        if args.heatmap_extractor is not None:
            fcnn = api.make_fcnn(args.heatmap_extractor)
            if args.verbose:
                print("Preprocessing images and exctracting features for subsequent training. This may take a while... "
                      "(+- 2 seconds per image)")

        # Define function to copy files
        def copy_files(file_list, split_type):
            """
            Copies samples in 'file_list' into a 'split_type' subdirectory. Puts .jpg images in an 'images' folder
            and corresponding .txt annotation files into a 'labels' folder. If heatmap_extractor is not None
            (high-precision training), it will copy the images' saliency maps into the target subdirectory.

            Args:
                file_list (list[str]): List of filenames, can be either jpg images or txt annotations.
                split_type (str): Name of target subdirectory (typically 'train' and 'val')

            Returns:
                None
            """
            images_target = os.path.join(dataset_dir, split_type, "images")
            labels_target = os.path.join(dataset_dir, split_type, "labels")
            os.makedirs(images_target, exist_ok=True)
            os.makedirs(labels_target, exist_ok=True)

            for filename in file_list:
                base = os.path.splitext(filename)[0]
                image_src = os.path.join(original_images_dir, filename)
                label_src = os.path.join(original_labels_dir, base + ".txt")

                image_dst = os.path.join(images_target, filename)
                label_dst = os.path.join(labels_target, base + ".txt")

                # If high-precision training, compute saliency maps of images
                if args.heatmap_extractor is not None:
                    cnn_full_img_res = 2016
                    img = Image.open(image_src)
                    img_array = np.array(img, dtype=np.float32)
                    img_shape = img_array.shape[:-1]
                    factor = np.max(img_array.shape) / cnn_full_img_res if np.max(img_array.shape) > cnn_full_img_res else 1
                    img_array = tf.image.resize(img_array,
                                                (int(img_array.shape[0] / factor), int(img_array.shape[1] / factor)),
                                                method=tf.image.ResizeMethod.BILINEAR)

                    preds = fcnn.predict(np.expand_dims(img_array, axis=0), verbose=1 if args.verbose else 0)
                    preds = np.squeeze(preds)

                    heatmap = training_api.extract_heatmap(img_array, np.expand_dims(preds[:, :, 1], axis=-1))
                    heatmap = tf.image.resize(heatmap, img_shape, method=tf.image.ResizeMethod.BILINEAR)
                    cv2.imwrite(image_dst, heatmap.numpy())
                # Else, simply copy the image
                else:
                    shutil.copy2(image_src, image_dst)
                # Copy label
                shutil.copy2(label_src, label_dst)

        # Copy files to new structure
        copy_files(train_files, "train")
        copy_files(val_files, "val")

        # Get absolute paths for yaml
        train_images_abs = os.path.abspath(os.path.join(dataset_dir, "train", "images"))
        val_images_abs = os.path.abspath(os.path.join(dataset_dir, "val", "images"))

        # Write the YAML file
        data_yaml = {
            "train": train_images_abs,
            "val": val_images_abs,
            "nc": len(class_names),
            "names": class_names
        }

        with open(output_yaml_path, "w") as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        if args.verbose:
            print(f"✅ Dataset prepared and YAML saved to: {output_yaml_path}")

    # Train the detector using dynamic multi-phase fine-tuning
    if not args.classification_only:
        # Prepare log directories
        api.warn_user_if_directory_exists("runs", silent=args.replace_all)
        api.warn_user_if_directory_exists("attempts", silent=args.replace_all)

        # Run fine_tune_yolo.py
        command = (
            f"python fine_tune_yolo.py --model {args.model} "
            f"--dataset {args.dataset} --epochs {args.epochs} --img_size {args.img_size} "
            f"--batch_init {args.batch_init} --batch_min {args.batch_min} --batch_decrease_rate {args.batch_decrease_rate} "
            f"--patience {args.patience} --fine_tuning_steps {args.fine_tuning_steps} "
            f"--lr0 {args.lr0} --lr_final {args.lr_final} --gpu {args.gpu}"
        )
        print(f"Running: {command}")
        try:
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with return code {e.returncode}")
            sys.exit(1)  # Stops the main script with a non-zero exit code

    # Select best model from the training 'runs' directory
    best_model = None
    # If classification-only, a pre-trained detector must be specified
    if args.classification_only:
        best_model = args.model
    else:
        # Select best based on map50
        best_map50 = 0
        run_dir = os.path.join("runs", "detect")
        for train_dir in os.listdir(run_dir):
            results_path = os.path.join(run_dir, train_dir, "results.csv")
            df = pd.read_csv(results_path)
            best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
            map50 = best_row['metrics/mAP50(B)']
            if best_map50 < map50:
                best_map50 = map50
                best_model = os.path.join(run_dir, train_dir, "weights", "best.pt")

        # Store best in 'output.pt'
        api.warn_user_if_file_exists("output.pt", silent=args.replace_all)
        shutil.copy2(best_model, "output.pt")
        if args.verbose:
            print("Best model ", best_model,
                  "achieved map50", best_map50, ". It was copied into output.pt")

    # Train post-detection binary classifier
    if not args.detection_only:
        # Prepare output file
        api.warn_user_if_file_exists("output.keras", silent=args.replace_all)

        # Run inference on training images to get true and false predictions
        original_argv = list(sys.argv)
        inference_args = f"inference_pipeline.py --input_folder {os.path.join(args.dataset, 'train', 'images')} " \
                         f"--model {best_model} --conf 0.01 --img_size {args.img_size} " \
                         f"--detection_only --write_conf --silent".split()
        sys.argv = list(inference_args)
        if args.verbose:
            print("Inferring on training dataset, may take a few seconds...")
        inference_args = inference_pipeline.parse_args()
        inference_pipeline.main(inference_args)

        # Store and classify true and false positives into a 'classify/train' directory
        api.warn_user_if_directory_exists("classify", silent=args.replace_all)
        os.makedirs(os.path.join("classify", "train"))
        if args.verbose:
            print("Storing true and false positives for posterior classification training...")
        training_api.save_tps_and_fps(os.path.join(args.dataset, 'train', 'images'), "output",
                                      os.path.join(args.dataset, 'train', 'labels'), os.path.join("classify", "train"),
                                      resize_mode=args.resize_mode)

        # Run inference on validation images to get true and false predictions
        inference_args = f"inference_pipeline.py --input_folder {os.path.join(args.dataset, 'val', 'images')} " \
                         f"--model {best_model} --conf 0.01 --img_size {args.img_size} " \
                         f"--detection_only --write_conf --silent".split()
        sys.argv = list(inference_args)
        if args.verbose:
            print("Inferring on validation dataset, may take a few seconds...")
        inference_args = inference_pipeline.parse_args()
        inference_pipeline.main(inference_args)

        # Store and classify true and false positives into a 'classify/val' directory
        if args.verbose:
            print("Storing true and false positives for posterior classification validation...")
        training_api.save_tps_and_fps(os.path.join(args.dataset, 'val', 'images'), "output",
                                      os.path.join(args.dataset, 'val', 'labels'), os.path.join("classify", "val"),
                                      resize_mode=args.resize_mode)


        # Load predictions and their corresponding labels (0 : fp, 1 : tp) into numpy arrays
        if args.verbose:
            print("Loading cropped bboxes for classification training...")
        X_train, y_train = training_api.make_image_and_label_array(os.path.join("classify", "train"))
        X_val, y_val = training_api.make_image_and_label_array(os.path.join("classify", "val"))
        y_train = tfk.utils.to_categorical(y_train, num_classes=2)
        y_val = tfk.utils.to_categorical(y_val, num_classes=2)
        train_label_ratios = np.sum(y_train, axis=0) / len(y_train)
        val_label_ratios = np.sum(y_val, axis=0) / len(y_val)

        # Plot 10 random images
        # labels_txt = ["false positive", "true positive"]
        # random_indices = np.random.choice(np.arange(len(X_train)), size=10, replace=False)
        # training_api.plot_images(X_train[random_indices], [labels_txt[int(np.argmax(y_train,axis=-1)[x])] for x in random_indices])
        # random_indices = np.random.choice(np.arange(len(X_val)), size=10, replace=False)
        # training_api.plot_images(X_val[random_indices], [labels_txt[int(np.argmax(y_val,axis=-1)[x])] for x in random_indices])


        np.random.seed(seed)

        # Undersample for a smaller dataset, if the hardware is struggling
        max_number = args.max_nb_images
        keep_ratio = max_number / len(y_train)
        if keep_ratio < 1 and max_number != -1:
            if args.verbose:
                print(f"Classifier training set too large, undersampling to max size {max_number}...")
            remaining_imgs = args.max_nb_images
            val_ratio = len(y_val) / (len(y_val) + len(y_train))
            remaining_imgs_val = int(val_ratio * remaining_imgs)
            remaining_imgs_train = remaining_imgs
            random_indices = np.random.choice(np.arange(len(y_val)), size=remaining_imgs_val, replace=False)
            random_indices = np.sort(random_indices)
            X_val, y_val = X_val[random_indices], y_val[random_indices]
            random_indices = np.random.choice(np.arange(len(y_train)), size=remaining_imgs_train, replace=False)
            random_indices = np.sort(random_indices)
            X_train, y_train = X_train[random_indices], y_train[random_indices]

        # Undersampling to achieve target ratio in training set
        if args.tp_ratio != -1:
            if args.verbose:
                print(f"Undersampling to achieve target ratio {args.tp_ratio}...")
            target_ratio = [1 - args.tp_ratio, args.tp_ratio]
            nfp_train = int(train_label_ratios[0] * len(y_train))
            ntp_train = len(y_train) - nfp_train
            if train_label_ratios[0] > target_ratio[0]:
                indices_nfp_train = np.random.permutation(nfp_train)
                X_train[:nfp_train] = X_train[indices_nfp_train]
                y_train[:nfp_train] = y_train[indices_nfp_train]
                to_remove = int(nfp_train - (target_ratio[0]/target_ratio[1]) * ntp_train)
                X_train = X_train[to_remove:]
                y_train = y_train[to_remove:]
            elif train_label_ratios[1] > target_ratio[1]:
                indices_ntp_train = np.random.permutation(ntp_train) + nfp_train
                X_train[nfp_train:] = X_train[indices_ntp_train]
                y_train[nfp_train:] = y_train[indices_ntp_train]
                to_remove = int(ntp_train - (target_ratio[1] / target_ratio[0]) * nfp_train)
                X_train = X_train[:-to_remove]
                y_train = y_train[:-to_remove]
            train_label_ratios = np.sum(y_train, axis=0) / len(y_train)

        # Create a permutation of indices
        indices_train = np.random.permutation(len(X_train))
        X_train = X_train[indices_train]
        y_train = y_train[indices_train]
        indices_val = np.random.permutation(len(X_val))
        X_val = X_val[indices_val]
        y_val = y_val[indices_val]

        if args.verbose:
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val length: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"Train Label Ratios (0: fp, 1: tp): {train_label_ratios}")
            print(f"Validation Label Ratios (0: fp, 1: tp): {val_label_ratios}")

        # Build conv network
        model = training_api.build_convnet(learning_rate=args.lr_classification)
        if args.verbose:
            model.summary()

        # Train conv network
        conv_history = model.fit(
            x=X_train,  # We need to apply the preprocessing thought for the ConvNeXt network, which is nothing
            y=y_train,
            # class_weight=class_weight_dict,
            batch_size=args.batch_classification,
            epochs=args.epochs_classification,
            validation_data=(X_val, y_val),  # We need to apply the preprocessing thought for the ConvNeXt network
            callbacks=[
                tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.patience_classification, restore_best_weights=True)]
        ).history

        # Plot the transfer learning and the fine-tuned ConvNeXt training histories
        # plt.figure(figsize=(15, 5))
        # plt.plot(conv_history['loss'], label='training', alpha=.3, color='#ff7f0e', linestyle='--')
        # plt.plot(conv_history['val_loss'], label='validation', alpha=.8, color='#ff7f0e')
        #
        # plt.legend(loc='upper left')
        # plt.title('Binary Crossentropy')
        # plt.grid(alpha=.3)
        #
        # plt.figure(figsize=(15, 5))
        # plt.plot(conv_history['accuracy'], label='training', alpha=.3, color='#ff7f0e', linestyle='--')
        # plt.plot(conv_history['val_accuracy'], label='validation', alpha=.8, color='#ff7f0e')
        # plt.legend(loc='upper left')
        # plt.title('Accuracy')
        # plt.grid(alpha=.3)
        #
        # plt.show()

        # Store classifier into output file
        model.save("output.keras")


def parse_args():
    parser = argparse.ArgumentParser(description="python training_pipeline.py --dataset my_dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset, in standard yolo format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="Gpu to use, the default is the macos standard (default: cuda:0)"
    )
    parser.add_argument(
        "--fine_tuning_steps",
        type=int,
        default=23,
        help="Number of fine-tuning runs, with a constant rate of layer "
             "unfreezing down to 0 frozen layers (default: 23)"
    )
    parser.add_argument(
        "--lr_classification",
        type=float,
        default=0.001,
        help="Learning rate for classification (default: 0.001)"
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
        help="Final run learning rate (default: lr0/100)"
    )
    parser.add_argument(
        "--batch_classification",
        type=int,
        default=32,
        help="Batch size for classification (default: 32)"
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
        default=16,
        help="Minimal batch size from which it will start plateauing (default: 16)"
    )
    parser.add_argument(
        "--batch_decrease_rate",
        type=float,
        default=-1,
        help="Rate at which batch size is reduced for each subsequent run "
             "(default: (batch_init - batch_min) / (fine_tuning_steps - 1))"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximal number of epochs per run (default: 20)"
    )
    parser.add_argument(
        "--epochs_classification",
        type=int,
        default=10,
        help="Number of epochs for classification (default: 10)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience fo early stopping (default: 5)"
    )
    parser.add_argument(
        "--patience_classification",
        type=int,
        default=5,
        help="Patience fo early stopping for the classifier (default: 5)"
    )
    parser.add_argument(
        "--tp_ratio",
        type=float,
        default=0.8,
        help="Ratio of true positives wished in training set for classification "
             "(default:0.8 for good true positive representation, but you may try "
             "values between [0-1] if model does not converge, or -1 if you don't want undersampling)"
    )
    parser.add_argument(
        "--max_nb_images",
        type=int,
        default=5000,
        help="Maximum size for the training set of the classifier (default: 5000). Usually the default is enough "
             "to train a strong classifier, but depending on your hardware you could change it or set it to -1 "
             "if you don't want any undersampling"
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
        "--verbose",
        action="store_true",
        help="Enable verbose output"
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
        "--no_split",
        action="store_true",
        help="In case the input folder is already split for validation "
             "and in yolo yaml format, this prevents further splitting"
    )
    parser.add_argument(
        "--replace_all",
        action="store_true",
        help="When set to True, script does not ask user to delete the previous output files, "
             "it deletes them by default (make sure you don't need those or have made a copy of "
             "them before specifying this flag)"
    )
    parser.add_argument(
        "--heatmap_extractor",
        type=str,
        default=None,
        help="Preprocess all input images with the feature extractor given as parameter, "
             "in order to train a high-precision and low-recall model (default: no preprocessing)"
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="pad",
        help="Mode of resizing of the bounding box for classification training. (default: \'pad\', "
             "mode \'bilinear\' will allow faster inference time at the cost of lower classification accuracy)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)