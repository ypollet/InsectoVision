import argparse
import numpy as np
from math import pow

# Single-class, change if your problem is multi-class
number_of_classes = 1
# Output heights of each YOLOv8 layer
output_heights = np.asarray([320, 160, 160,
                       80, 80, 40,
                       40, 20, 20,
                       20, 40, 40,
                       40, 80, 80,
                       80, 40, 40,
                       40, 20, 20,
                       20])
# Output widths
output_widths = np.copy(output_heights)
# Output channels
output_channels = np.asarray([64, 128, 128,
                       256, 256, 512,
                       512, 512, 512,
                       512, 512, 512,
                       512, 512, 768,
                       256, 256, 768,
                       512, 512, 512,
                       512])
# Factors to apply to each layer when converting to another model size (default: ones for YOLOv8l)
factors = np.ones(22)

# For every layer and sub-layer in YOLOv8, the following methods
# compute the feature map dimensions needed from their outputs
def conv2d(h, w, c):
    return h*w*c
def upsample(h, w, c):
    return h*w*c
def concat(h, w, c):
    return h*w*c
def silu(h, w, c):
    return h*w*c
def batchnorm(h, w, c):
    # Feature map size including moving average and scale/shift (h * w + 2) * c
    return (h*w+2)*c
def maxpool(h,w,c):
    # Feature map size including the 'indices' map for backward propagation (same size as output)
    return 2*h*w*c
def conv(h, w, c):
    # Conv is conv2d + batchnorm + silu
    return conv2d(h,w,c) + batchnorm(h,w,c) + silu(h,w,c)
def bottleneck(h,w,c):
    # Bottleneck is 2 conv
    return 2*conv(h,w,c)
def c2f(h,w,c,n):
    # C2f is 2 conv and n bottlenecks with half the number of channels
    return 2*conv(h,w,c) + n*bottleneck(h,w,c/2)
def c2f3(h,w,c,d):
    # C2f3 is c2f with 3*d bottlenecks, with d the network depth ratio
    return c2f(h,w,c,int(3*d))
def c2f6(h,w,c,d):
    # C2f6 is c2f with 6*d bottlenecks, with d the network depth ratio
    return c2f(h,w,c,int(6*d))
def sppf(h,w,c):
    # Sppf is 2 conv and 3 maxpool
    return 2*conv(h,w,c) + 3*maxpool(h,w,c)
def detect(h,w,c, factor, w_factor, r_factor, nc):
    # Here, h-w-c are the input dimensions.
    # Each detection head has 2 conv blocks.
    # Factor at the last conv2d layer should not include r_factor and w_factor,
    # because output feature maps have respectively 4*reg_max and nc channels
    return 4*conv(h,w,c)*factor + h*w*(4*16 + nc)*(factor/(w_factor*r_factor))


# Layers of YOLOv8
layers = [conv, conv, c2f3, conv, c2f6, conv, c2f6, conv, c2f3,
          sppf, upsample, concat, c2f3, upsample, concat, c2f3,
          conv, concat, c2f3, conv, concat, c2f3]
# Boolean array specifying if the corresponding layer in 'layers' is c2f
is_c2f = [(x is c2f3 or x is c2f6) for x in layers]


class YoloModel:
    # YOLO model representation to compute theoretical memory needs
    def get_factors(self):
        my_factors = np.copy(factors)
        # Multiply by the relevant factors to get the right model size (n, s, m, l, x)
        my_factors[7] = self.r
        my_factors[8] = self.r
        my_factors[9] = self.r
        my_factors[10] = self.r
        my_factors[11] = (1 + self.r)
        my_factors[20] = (1 + self.r)
        my_factors *= self.w
        # Convert to given input image size
        my_factors *= pow((self.img_size / 640), 2)
        # Convert to given floating-point precision
        my_factors *= int(self.fp/8)
        return my_factors

    def __init__(self, d, w, r, img_size=640, fp=32, nc=1):
        # Build YOLOModel object with the factors and parameters of your model configuration
        self.d = d
        self.w = w
        self.r = r
        self.img_size = img_size
        self.fp = fp
        self.nc = nc
        self.fact = self.get_factors()

    def get_mem_needed(self):
        # Compute the total memory needs of your model to store feature maps
        acc = 0
        for i, tup in enumerate(zip(is_c2f, layers)):
            include_d, l = tup
            if include_d:
                acc += l(output_widths[i], output_heights[i],
                         output_channels[i], self.d) * self.fact[i]
            else:
                acc += l(output_widths[i], output_heights[i],
                         output_channels[i]) * self.fact[i]
        # Add detection heads
        acc += detect(output_widths[15], output_heights[15],
                      output_channels[15], self.fact[15],
                      self.w, 1, self.nc)
        acc += detect(output_widths[18], output_heights[18],
                      output_channels[18], self.fact[18],
                      self.w, 1, self.nc)
        acc += detect(output_widths[21], output_heights[21],
                      output_channels[21], self.fact[21],
                      self.w, self.r, self.nc)
        return acc


# Names, factors and parameters for each YOLO model size
model_names = ["n", "s", "m", "l", "x"]
model_names = ["YOLOv8" + x for x in model_names]
model_factors = [(1/3, 1/4, 2), (1/3, 1/2, 2),
               (2/3, 3/4, 1.5), (1, 1, 1),
                 (1, 5/4, 1)]
model_params = [3.2, 11.2, 25.9, 43.7, 68.2]


def find_optimal_config(batch, fp, nc, ram, system_ram):
    """
    Determines the highest viable input resolution for each YOLO model configuration
    that fits within the available RAM constraints.

    Parameters:
        batch (int): Batch size used during training/inference.
        fp (int): Floating point precision (e.g., 16 or 32 bits).
        nc (int): Number of classes in the dataset.
        ram (float): Total available GPU RAM in gigabytes.
        system_ram (float): Reserved GPU RAM for system overhead in gigabytes.

    Returns:
        None: Prints the optimal resolution per model, or a message if training is not feasible.
    """

    # Define the list of candidate image resolutions (from 320 to 2560 in steps of 320).
    img_sizes = list(range(320, 2561, 320))
    # Compute usable RAM by subtracting system overhead, then convert to bytes.
    RAM = ram - system_ram
    RAM *= 1000000000

    # Iterate over each model architecture defined globally.
    for i in range(len(model_names)):
        name = model_names[i] # Model name (e.g., 'YOLOv8n', 'YOLOv8m', etc.)
        args = model_factors[i] # Architecture scaling factors (depth, width, resolution factor)
        params = model_params[i] # Number of model parameters in millions

        best_res, best_mem = 0, 0
        # Try each image size and see which one fits in memory
        for s in img_sizes:

            d, w, r = args
            model = YoloModel(d, w, r, s, fp, nc) # Create a model instance with current config
            mem = model.get_mem_needed()*batch # Memory to store feature maps
            mem += params*1000000*4*int(fp/8) # Memory to store params
            mem += batch*int(fp/8)*s*s*3 # Memory to store batch

            # Keep the configuration that maximizes resolution without exceeding RAM
            if 0 < RAM - mem < RAM - best_mem:
                best_res = s
                best_mem = mem

        # Report result for the model
        if best_res == 0:
            print(f"No training possible with model {name}")
        else:
            best_mem_gb = best_mem/1000000000
            print(f"Optimal resolution for model {name} is {best_res} "
                  f"which uses {best_mem_gb} GBytes")


def main(args):
    # Case 1: Direct memory estimation for a given model config
    if not args.find_opti:
        # Find the index, name, factors and params of the selected model (e.g., 'YOLOv8n', 'YOLOv8m', etc.)
        idx = model_names.index("YOLOv8" + args.model_size)
        name = model_names[idx]
        model_args = model_factors[idx]
        params = model_params[idx]

        # Retrieve input arguments
        batch = args.batch
        fp = args.fp
        nc = args.nc
        d, w, r = model_args
        res = args.img_size

        # Create a YOLO model instance with the specified configuration, and compute memory needs per image
        model = YoloModel(d, w, r, img_size=res, fp=fp, nc=nc)
        mem = model.get_mem_needed() * batch
        mem_1img = mem / (batch * 1000000000)
        print(f"Memory needed to store the {name} feature maps of a single {res}*{res} image, "
              f"with precision {fp} : {mem_1img} GBytes")

        # Compute total memory usage
        mem += params * 1000000 * 4 * int(fp / 8)  # mem to store params
        mem += batch * int(fp / 8) * res * res * 3  # mem to store batch
        mem_gb = mem/1000000000
        print(f"Memory needed to train {name} on {res}*{res} images, "
              f"batch {batch} and precision {fp} : {mem_gb} GBytes")

    # Case 2: Search for optimal training resolution that fits within memory
    if args.find_opti:
        # Ensure user has specified total GPU RAM
        if args.ram is None:
            raise ValueError("Please specify your RAM if "
                             "you want to search for optimal "
                             "training configuration")
        else:
            # Call helper to determine the best resolution for each model
            find_optimal_config(args.batch, args.fp, args.nc,
                                args.ram, args.system_ram)


def parse_args():
    parser = argparse.ArgumentParser(description="python memory_analysis.py --find_opti --ram 8")

    parser.add_argument(
        "--model_size",
        type=str,
        default="l",
        help="YOLO size (n, s, m, l or x)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Image resolution"
    )
    parser.add_argument(
        "--fp",
        type=int,
        default=32,
        help="Floating point precision (default: 32)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="Number of classes (default: 1)"
    )
    parser.add_argument(
        "--find_opti",
        action="store_true",
        help="Allows grid search for best training params. Arg "
             "--ram must be specified"
    )
    parser.add_argument(
        "--ram",
        type=int,
        default=None,
        help="machine's RAM in GB"
    )
    parser.add_argument(
        "--system_ram",
        type=float,
        default=1.5,
        help="system-allocated RAM in GB which cannot be "
             "used for training (default: 1.5)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)