"""
python test/test_load_h5.py --h5-path $HOME/datasets/episode_0.hdf5
"""

import tyro
import h5py
import cv2
from pathlib import Path
import numpy as np


def read_hdf5(h5_path: Path, print_structure: bool = True, print_data: bool = True):
    """
    Read an HDF5 file and print its structure and data.

    Args:
        h5_path (Path): Path to the HDF5 file
        print_structure (bool): Whether to print the file structure
        print_data (bool): Whether to print the data content
    """
    try:
        with h5py.File(h5_path, "r") as f:
            print(f"Successfully opened file: {h5_path}")
            h5_name = []
            # Print file structure
            if print_structure:
                print("\nFile structure:")

                def print_attrs(name, obj):
                    print(f"  {name} (Type: {type(obj)})")
                    h5_name.append(name)
                    if isinstance(obj, h5py.Dataset):
                        print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                    for key, val in obj.attrs.items():
                        print(f"    Attribute: {key} = {val}")

                f.visititems(print_attrs)

            # Print data content (only partial data to avoid excessive output)
            if print_data:
                for name in h5_name:
                    dataset = f[name]
                    print(f"\nData from {name}:")

                    if isinstance(dataset, h5py.Dataset):
                        # Get dataset shape and type
                        shape = dataset.shape
                        dtype = dataset.dtype
                        print("------------------------------------------------")
                        print(f"Shape: {shape}, Dtype: {dtype}")
                        # Print memory usage
                        print(f"\nTotal dataset size: {dataset.size * dataset.dtype.itemsize / (1024**2):.2f} MB")

                        # Special handling for datasets with name containing 'observations/images'
                        if "observations/images" in name:
                            print("Image Dataset [width, height, channels]:")
                            print(dataset[0].dtype)
                            # Print statistics
                            sample = (
                                dataset[0]
                                if dataset[0].dtype == "uint8"
                                else cv2.imdecode(np.frombuffer(dataset[0], dtype=np.uint8), cv2.IMREAD_COLOR)
                            )

                            cv2.imwrite(f"{name.split('/')[-1]}.jpg", sample)
                            print(f"Sample image shape: {sample.shape}")
                            print(f"Pixel value range: {sample.min()} - {sample.max()}")

                            # Print corner pixels (top-left 4x4 area)
                            print("\nTop-left 4x4 corner of first image (R,G,B channels):")
                            print(sample[:4, :4, :])

                            # Print center pixel values
                            center_y, center_x = sample.shape[0] // 2, sample.shape[1] // 2
                            print("\nCenter pixel values (R,G,B):")
                            print(sample[center_y, center_x, :])

                        else:
                            # Standard data printing for non-image datasets
                            data = dataset[...]
                            if data.size > 10:
                                print("First 10 elements:", data.flatten()[:10], "...")
                            else:
                                print(data)
                    else:
                        print(f"{name} is not a dataset.")

    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    tyro.cli(read_hdf5)
