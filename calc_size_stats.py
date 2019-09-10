import cv2
import numpy as np
from pathlib import Path
import pandas as pd

root_path = Path("data/well1")
dir_list = train_list = list(range(1, 31)) + list(range(51, 131))
dir_tmpl = "Seq0000_T1_XY{:03d}"

crop_sizes = [48, 56, 100, 128]

for crop_sz in crop_sizes:
    n_cells = []
    for d in dir_list:
        sd = root_path / dir_tmpl.format(d)

        mask = cv2.imread(str(sd / "Labelled_cells.tiff"), cv2.CV_16U)

        h, w = mask.shape

        df = pd.read_csv(sd / "Cells_rawints.csv")[["ObjectNumber",
                                                    "Location_Center_X",
                                                    "Location_Center_Y"]]
        df = df[~df.isnull().any(axis=1)]

        df["left"] = df["Location_Center_X"].astype(int) - crop_sz // 2
        df["top"] = df["Location_Center_Y"].astype(int) - crop_sz // 2
        df["right"] = df["left"] + crop_sz
        df["bottom"] = df["top"] + crop_sz
        df = df[(df["top"] >= 0) & (df["left"] >= 0) & (df["bottom"] <= h) & (df["right"] <= w)]

        for _, row in df.iterrows():
            top, bottom, left, right = row[["top", "bottom", "left", "right"]].astype(int).values
            crop = mask[top: bottom, left: right]
            n_cells.append(len([i for i in np.unique(crop) if i != 0]))

    n_cells = np.array(n_cells)
    print("Crop {}, contained cells {}, standard deviation {}".format(crop_sz, n_cells.mean(), n_cells.std()))