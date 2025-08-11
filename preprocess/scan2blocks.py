""" Split scan into blocks
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import glob
import numpy as np
from util.logger import get_logger

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR SUPERPOINT GRAPH GENERATION
# -----------------------------------------------------------------------------


def get_last_processed_scan(save_path):
    """Find the latest modified file in the save_path/data directory and parse the scanname from it.
    
    Args:
        save_path: Path to the save directory
        
    Returns:
        last_scan_name: Name of the last processed scan, or None if no files exist
    """
    data_dir = os.path.join(save_path, "data")
    if not os.path.exists(data_dir):
        return None
    
    # Get all .npy files in the data directory
    block_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not block_files:
        return None
    
    # Find the file with the latest modification time
    latest_file = max(block_files, key=os.path.getmtime)
    
    # Parse scan name from filename (format: scanname_block_X.npy)
    filename = os.path.basename(latest_file)
    if "_block_" in filename:
        scan_name = filename.split("_block_")[0]
        return scan_name
    
    return None


def filter_file_paths(file_paths, last_processed_scan):
    """Filter file_paths to only include files from the last processed scan onwards.
    
    Args:
        file_paths: List of file paths to process
        last_processed_scan: Name of the last processed scan
        
    Returns:
        filtered_paths: List of file paths from the last processed scan onwards
    """
    if last_processed_scan is None:
        return file_paths
    
    # Find the index of the last processed scan
    last_scan_index = -1
    for i, file_path in enumerate(file_paths):
        scan_name = os.path.basename(file_path)[:-4]  # Remove .npy extension
        if scan_name == last_processed_scan:
            last_scan_index = i
            break
    
    if last_scan_index == -1:
        # If we can't find the last processed scan, return all files
        return file_paths
    
    # Return files from the last processed scan onwards (including it)
    return file_paths[last_scan_index:]


def scan2blocks(data, block_size, stride, min_npts):

    """Prepare block data.
    Args:
        data: N x 7 numpy array, 012 are XYZ in meters, 345 are RGB in [0,255], 6 is the labels
            assumes the data is not shifted (min point is not origin),
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
    Returns:
        blocks_list: a list of blocks, each block is a num_point x 7 np array
    """
    logger = get_logger(name="scan2blocks")
    print(type(stride))
    print(type(block_size))
    assert stride <= block_size

    xyz = data[:, :3]
    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    xyz_max = np.amax(xyz, axis=0)

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((xyz_max[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((xyz_max[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride)
            ybeg_list.append(j * stride)

    # Collect blocks
    blocks_list = []
    logger.info(f"Processing {len(xbeg_list)} blocks")
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (xyz[:, 0] <= xbeg + block_size) & (xyz[:, 0] >= xbeg)
        ycond = (xyz[:, 1] <= ybeg + block_size) & (xyz[:, 1] >= ybeg)
        cond = xcond & ycond
        if(np.sum(cond)<min_npts):
            logger.info(f"Skipping block {idx} because it has less than {min_npts} points \t {len(blocks_list)}/{len(xbeg_list)}")
            continue
        if (
            np.all(data[cond, 6] == 0)
        ):  # discard block if there are less than 100 pts.
            logger.info(f"Skipping block {idx} because it has only class 0 \t {len(blocks_list)}/{len(xbeg_list)}")
            continue

        block = data[cond, :]
        blocks_list.append(block)
        logger.info(f"Added block {idx} with {np.sum(cond)} points")

    return blocks_list


def scan2blocks_wrapper(scan_path, block_size, stride, min_npts):
    if scan_path[-3:] == "txt":
        data = np.loadtxt(scan_path)
    elif scan_path[-3:] == "npy":
        data = np.load(scan_path)
    else:
        print("Unknown file type! exiting.")
        exit()
    return scan2blocks(data, block_size, stride, min_npts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="[Preprocessing] Split scans into blocks"
    )
    parser.add_argument("--data_path", default="/sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/processed/")
    parser.add_argument(
        "--block_size",
        type=float,
        default=1,
        metavar="s",
        help="size of each block",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1,
        help="stride of sliding window for splitting scans, "
        "stride should be not larger than block size",
    )
    parser.add_argument(
        "--min_npts",
        type=int,
        default=1000,
        help="the minimum number of points in a block,"
        "if less than this threshold, the block is discarded",
    )

    args = parser.parse_args()

    DATA_PATH = args.data_path
    BLOCK_SIZE = args.block_size
    STRIDE = args.stride
    MIN_NPTS = args.min_npts
    SAVE_PATH = os.path.join(
        os.path.dirname(DATA_PATH),
        "blocks_bs{0}_s{1}".format(BLOCK_SIZE, STRIDE),
        "data",
    )
    
    # Check if SAVE_PATH already exists and find the last processed scan
    last_processed_scan = None
    if os.path.exists(SAVE_PATH):
        print(f"SAVE_PATH already exists: {SAVE_PATH}")
        last_processed_scan = get_last_processed_scan(SAVE_PATH)
        if last_processed_scan:
            print(f"Last processed scan: {last_processed_scan}")
        else:
            print("No processed files found, starting from beginning")
    else:
        os.makedirs(SAVE_PATH)
        print(f"Created new SAVE_PATH: {SAVE_PATH}")

    file_paths = glob.glob(os.path.join(DATA_PATH, "*.npy"))
    print("{} scans to be split...".format(len(file_paths)))
    
    # Filter file paths if resuming from a previous run
    if last_processed_scan:
        original_count = len(file_paths)
        file_paths = filter_file_paths(file_paths, last_processed_scan)
        remaining_count = len(file_paths)
        print(f"Resuming from after scan '{last_processed_scan}': {remaining_count}/{original_count} scans remaining")
    
    block_cnt = 0
    for file_path in file_paths:
        scan_name = os.path.basename(file_path)[:-4]
        blocks_list = scan2blocks_wrapper(
            file_path, block_size=BLOCK_SIZE, stride=STRIDE, min_npts=MIN_NPTS
        )
        print(
            "{0} is split into {1} blocks.".format(scan_name, len(blocks_list))
        )
        block_cnt += len(blocks_list)

        for i, block_data in enumerate(blocks_list):
            block_filename = scan_name + "_block_" + str(i) + ".npy"
            np.save(os.path.join(SAVE_PATH, block_filename), block_data)

    print("Total samples: {0}".format(block_cnt))
