import argparse
import os.path

import subsample
import api

strategy_names = ["random", "uniform", "uncertainty", "diversity", "supervised"]
strategies = [subsample.random_sample, subsample.uniform_partition, subsample.max_mean_uncertainty_sample, subsample.diverse_sample, subsample.supervised_sample]
seed = 69


def main(args):
    if args.strategy not in strategy_names:
        raise ValueError(f"Argument --strategy must be one of the following {strategy_names}")
    else:
        subsampling_strat = strategies[strategy_names.index(args.strategy)]

    if args.model is None and args.strategy not in strategy_names[:2]:
        raise ValueError(f"Argument --model cannot be None with strategy {args.strategy}, "
                         f"please specify a .pt model")

    if args.labeled is None and args.strategy not in strategy_names[:3]:
        raise ValueError(f"Argument --labeled cannot be None with strategy {args.strategy}, "
                         f"because this strategy needs the original training set. Please specify the path "
                         f"of your labeled dataset")

    indices = subsampling_strat(args.unlabeled, args.labeled, args.model, args.size, seed)
    api.make_selection_from_indices(args.output_selection, os.path.join(args.unlabeled, "images"), indices, silent=args.silent)
    api.make_selection_from_indices(args.output_remaining,  os.path.join(args.unlabeled, "images"),
                                    [i for i in range(len(api.get_images(os.path.join(args.unlabeled, "images")))) if i not in indices],
                                    silent=args.silent)


def parse_args():
    parser = argparse.ArgumentParser(description="python active_selection.py --labeled my_labeled_dataset "
                                                 "--unlabeled dataset_to_label --strategy supervised "
                                                 "--model my_model.pt")

    parser.add_argument(
        "--labeled",
        type=str,
        default=None,
        help="Path to the labeled dataset, in standard yolo format, with subfolders images and labels "
             "(default: None, only valid for strategies that do not need the original training set)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Subsampling strategy to use for image selection (default:uncertainty, must be one of the following: "
             f"{strategy_names})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pretrained detection model, used for active learning subsampling (default: None, but this "
             "default value is only valid for strategies random and uniform)"
    )
    parser.add_argument(
        "--unlabeled",
        type=str,
        required=True,
        help="Path to the unlabeled dataset, with subfolder images, to get an optimal subsample of images "
             "to annotate with active-learning techniques"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=30,
        help="Number of additional images to include in selection (default: 30)"
    )
    parser.add_argument(
        "--output_selection",
        type=str,
        default="selection",
        help="Path to the output directory where the selected images will be stored (default: selection)"
    )
    parser.add_argument(
        "--output_remaining",
        type=str,
        default="unlabeled",
        help="Path to the output directory where the remaining images, which were not selected, are stored. "
             "Use it as your unlabeled dataset in subsequent AL iterations ! (default: unlabeled)"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="When set to True, script does not warn and ask confirmation if existing files will be deleted"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)