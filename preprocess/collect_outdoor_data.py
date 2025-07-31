import os
import numpy as np

def map_street_sign_class(data):
    data[:, 9] = np.where(data[:, 8] == 15, data[:, 9], 0)
    return data
    
if __name__ == "__main__":
    import argparse

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

    txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    for txt_file in txt_files:
        data = np.loadtxt(os.path.join(DATA_PATH, txt_file), delimiter=' ', skiprows=1)
        data = map_street_sign_class(data)
        np.save(os.path.join(SAVE_PATH, txt_file.replace('.txt', '_processed.npy')), data)