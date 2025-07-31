import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.logger import get_logger

def map_street_sign_class(data):
    data[:, 6] = np.where(data[:, 6] == 15, data[:, 7], 0)
    data = np.delete(data, 7, axis=1)
    return data
    

if __name__ == "__main__":
    import argparse

    logger = get_logger(name="collect_outdoor_data")
    logger.info("Starting to collect outdoor data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/",
        help="Directory to dataset",
    )
    parser.add_argument(
        "--save_path",
        default="/sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/processed",
        help="Directory to save",
    )
    args = parser.parse_args()

    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]
    for txt_file in txt_files:
        logger.info(f"Processing {txt_file}")
        data = np.loadtxt(os.path.join(DATA_PATH, txt_file), delimiter=",", skiprows=1)
        data = map_street_sign_class(data)
        file_name = f"{txt_file.split('S')[0].strip()}_{txt_file.split('part')[-1].split('.')[0] if 'part' in txt_file else ''}_processed.npy".replace(" ", "")
        logger.info(f"Saving to {file_name} with shape {data.shape}")
        np.set_printoptions(suppress=True)
        np.save(
            os.path.join(SAVE_PATH, file_name), data
        )
