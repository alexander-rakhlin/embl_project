import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import threading
from keras import backend as K
from keras.utils import to_categorical

from models import m46c, m46r

np.random.seed(0)

bad_frames = [
    'Seq0000_T1_XY001',
    'Seq0000_T1_XY002',
    'Seq0000_T1_XY003',
    'Seq0000_T1_XY004',
    'Seq0000_T1_XY005',
    'Seq0000_T1_XY067',
    'Seq0000_T1_XY068',
    'Seq0000_T1_XY072',
    'Seq0000_T1_XY126',
    'Seq0000_T1_XY127',
    'Seq0000_T1_XY128',
    'Seq0000_T1_XY129',
    'Seq0000_T1_XY130',
]
bad_frames = [int(s[-3:]) for s in bad_frames]

root = Path("data/well1")
out_root = Path("data/processed/well1")

SZ = 48  # crop size
loss_balance = 0.5
epochs = 20
lr = 1e-3
batch_size = 32
target_columns = ["rawRed", "rawGreen"]
n_classes = 4
n_channels = 3

weights_path = out_root / "model20x20_cls{}_{}_chnl{}.hd5".format(n_classes, SZ, n_channels)
init_weights = None
descriptors_path = out_root / "descriptors20x20_cls{}_{}_chnl{}.pkl".format(n_classes, SZ, n_channels)

train_list = list(range(1, 31)) + list(range(51, 131))
val_list = list(range(31, 51))

train_list = [frame for frame in train_list if frame not in bad_frames]
val_list = [frame for frame in val_list if frame not in bad_frames]


def crop_cells():
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        print(sd.name)
        (out_root / sd.name).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(sd / "Cells.csv")[["ObjectNumber", "Location_Center_X", "Location_Center_Y"]]
        ixy = df[~df.isnull().any(axis=1)].astype(int).as_matrix()

        for ch in [0, 3, 4]:
            channel = cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U)
            for i, x, y in ixy:
                top, left = y - SZ // 2, x - SZ // 2
                h, w = channel.shape
                if top < 0 or left < 0 or (top + SZ) > h or (left + SZ) > w: continue
                cv2.imwrite(str(out_root / sd.name / "chan_{}_{}.tiff".format(ch, i)),
                            channel[top: top + SZ, left: left + SZ])


def mean_std():
    '''
    Channel 0, mean 14991, std 2112
    Channel 1, mean 235, std 435
    Channel 2, mean 190, std 82
    Channel 3, mean 1170, std 1557
    Channel 4, mean 532, std 353
    '''
    for ch in [0, 1, 2, 3, 4]:
        channels = []
        for sd in root.iterdir():
            if not sd.is_dir(): continue
            channels.append(cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U).flatten())
        channels = np.concatenate(channels)
        print("Channel {}, mean {}, std {}".format(ch, int(channels.mean()), int(channels.std())))


def make_targets():
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        print(sd.name)
        (out_root / sd.name).mkdir(parents=True, exist_ok=True)
        cells = cv2.imread(str(sd / "Labelled_cells.tiff"), cv2.CV_16U)

        channel1 = cv2.imread(str(sd / "chan_1.tiff"), cv2.CV_16U)
        channel2 = cv2.imread(str(sd / "chan_2.tiff"), cv2.CV_16U)
        y = {}
        for i in np.unique(cells):
            y1 = channel1[cells == i].mean()
            y2 = channel2[cells == i].mean()
            y[i] = (y1, y2)

        with open(out_root / sd.name / "y.pkl", "wb") as f:
            pickle.dump(y, f)


def add_intensities():
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        print(sd.name)
        cells = cv2.imread(str(sd / "Labelled_cells.tiff"), cv2.CV_16U)

        channel1 = cv2.imread(str(sd / "chan_1.tiff"), cv2.CV_16U)
        channel2 = cv2.imread(str(sd / "chan_2.tiff"), cv2.CV_16U)
        df = pd.read_csv(sd / "Cells.csv")
        df["rawRed"] = 0
        df["rawGreen"] = 0
        for i in df["ObjectNumber"]:
            mask = cells == i
            if mask.any():
                y1 = channel1[mask].mean()
                y2 = channel2[mask].mean()
            else:
                y1 = np.nan
                y2 = np.nan
            df.loc[df["ObjectNumber"] == i, "rawRed"] = y1
            df.loc[df["ObjectNumber"] == i, "rawGreen"] = y2
        df.to_csv(sd / "Cells_rawints.csv", index=False, na_rep="nan")


def add_intensities20():
    SZ = 20
    SZ2 = SZ // 2
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        print(sd.name)

        cy3 = cv2.imread(str(sd / "chan_1.tiff"), cv2.CV_16U)
        gfp = cv2.imread(str(sd / "chan_2.tiff"), cv2.CV_16U)
        df = pd.read_csv(sd / "Cells_rawints.csv")
        df["rawRed20"] = 0
        df["rawGreen20"] = 0
        for i in df["ObjectNumber"]:
            x, y = df.loc[df["ObjectNumber"] == i, ["Location_Center_X", "Location_Center_Y"]].values[0]
            if np.isnan(x) or np.isnan(y):
                continue
            x, y = max(SZ2, int(x)), max(SZ2, int(y))
            green_median = np.median(gfp[y - SZ2: y + SZ2, x - SZ2: x + SZ2])
            red_median = np.median(cy3[y - SZ2: y + SZ2, x - SZ2: x + SZ2])
            df.loc[df["ObjectNumber"] == i, "rawRed20"] = red_median
            df.loc[df["ObjectNumber"] == i, "rawGreen20"] = green_median
        df.to_csv(sd / "Cells_rawints20.csv", index=False, na_rep="nan")


def mean_std_target():
    """
    y1: mean 354, std 560
    y2: mean 222, std 83
    """
    y1, y2 = [], []
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        with open(out_root / sd.name / "y.pkl", "rb") as f:
            y = pickle.load(f)
        y1_, y2_ = list(zip(*list(y.values())))
        y1.extend(y1_)
        y2.extend(y2_)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y1 = y1[~np.isnan(y1)]
    y2 = y2[~np.isnan(y2)]
    print("y1: mean {}, std {}\n"
          "y2: mean {}, std {}".format(int(y1.mean()), int(y1.std()), int(y2.mean()), int(y2.std())))


def load_dir(d, dir_tmpl="Seq0000_T1_XY{:03d}"):
    sd = out_root / dir_tmpl.format(d)
    cells = sd.glob("chan_0_*.tiff")
    cells_n = sorted(int(c.name.split(".")[0].split("_")[-1]) for c in cells)
    with open(sd / "y.pkl", "rb") as f:
        labels = pickle.load(f)
    x = []
    y = []
    for n in cells_n:
        if not n in labels: continue
        sample = [cv2.imread(str(sd / "chan_{}_{}.tiff".format(ch, n)), cv2.CV_16U) for ch in [0, 3, 4]]
        x.append(np.array(sample))
        y.append(labels[n])
    x = np.array(x, dtype=K.floatx())
    y = np.array(y, dtype=K.floatx())

    ch0_mean, ch0_std = 14991, 2112
    ch3_mean, ch3_std = 1170, 1557
    ch4_mean, ch4_std = 532, 353
    x[:, 0] = (x[:, 0] - ch0_mean) / ch0_std
    x[:, 1] = (x[:, 1] - ch3_mean) / ch3_std
    x[:, 2] = (x[:, 2] - ch4_mean) / ch4_std
    if K.image_data_format() == "channels_last":
        x = np.moveaxis(x, 1, -1)
    y1_mean, y1_std = 354, 560
    y2_mean, y2_std = 222, 83
    y[:, 0] = (y[:, 0] - y1_mean) / y1_std
    y[:, 1] = (y[:, 1] - y2_mean) / y2_std

    return x, y


def create_dataset(dirs, output_f):
    data = [load_dir(d) for d in dirs]
    x, y = zip(*data)
    x = np.concatenate(x)
    y = np.concatenate(y)
    with open(output_f, "wb") as f:
        pickle.dump((x, y), f)


def train(train_path, test_path, weights_path="data/model.hd5", epochs=20, lr=1e-4):
    with open(train_path, "rb") as f:
        x_train, y_train = pickle.load(f)
    with open(test_path, "rb") as f:
        x_test, y_test = pickle.load(f)

    y1_mean, y1_std = 354, 560
    y2_mean, y2_std = 222, 83
    y_train[:, 0] = y_train[:, 0] * y1_std + y1_mean
    y_train[:, 1] = y_train[:, 1] * y2_std + y2_mean
    y_test[:, 0] = y_test[:, 0] * y1_std + y1_mean
    y_test[:, 1] = y_test[:, 1] * y2_std + y2_mean

    # y_train[:, 0] = y_train[:, 0] * 40 + y1_mean
    # y_test[:, 0] = y_test[:, 0] * 40 + y1_mean
    # y_train[:, 1] = y_train[:, 1] * y2_std + y2_mean
    # y_test[:, 1] = y_test[:, 1] * y2_std + y2_mean

    model = m46r(include_top=True, lr=lr)
    model.fit(x_train, [y_train[:, 0], y_train[:, 1]],
              validation_data=(x_test, [y_test[:, 0], y_test[:, 1]]),
              epochs=epochs, batch_size=32)
    model.save_weights(weights_path)


def predict(test_path, weights_path, descriptors_path):
    with open(test_path, "rb") as f:
        x, y = pickle.load(f)

    y1_mean, y1_std = 354, 560
    y2_mean, y2_std = 222, 83
    y[:, 0] = y[:, 0] * y1_std + y1_mean
    y[:, 1] = y[:, 1] * y2_std + y2_mean

    model = m46r(include_top=False, weights=weights_path)
    descriptors = model.predict(x)
    with open(descriptors_path, "wb") as f:
        pickle.dump((descriptors, y), f)


def preprocess_x(x):
    x_cp = x.copy()
    ch0_mean, ch0_std = 14991, 2112
    ch3_mean, ch3_std = 1170, 1557
    ch4_mean, ch4_std = 532, 353
    x_cp[:, 0] = (x_cp[:, 0] - ch0_mean) / ch0_std
    x_cp[:, 1] = (x_cp[:, 1] - ch3_mean) / ch3_std
    x_cp[:, 2] = (x_cp[:, 2] - ch4_mean) / ch4_std
    if K.image_data_format() == "channels_last":
        x_cp = np.moveaxis(x_cp, 1, -1)
    return x_cp


def preprocess_x_(x):
    x_cp = x.copy()
    ch0_mean, ch0_std = 14991, 2112
    ch3_mean, ch3_std = 1170, 1557
    ch4_mean, ch4_std = 532, 353
    x_cp[:, 0] = np.log(x_cp[:, 0]) - np.log(ch0_mean) - np.log(ch0_std)
    x_cp[:, 1] = np.log(x_cp[:, 1]) - np.log(ch3_mean) - np.log(ch3_std)
    x_cp[:, 2] = np.log(x_cp[:, 2]) - np.log(ch4_mean) - np.log(ch4_std)
    if K.image_data_format() == "channels_last":
        x_cp = np.moveaxis(x_cp, 1, -1)
    return x_cp


def preprocess_y(y):
    y_cp = y.copy()
    return [y_cp[:, 0], y_cp[:, 1]]


def aux_filter_rectangular(df):
    fun = lambda x: np.log10(1 + x)
    m = np.median(fun(df[["rawRed", "rawGreen"]]), axis=0)
    s = np.std(fun(df[["rawRed", "rawGreen"]]), axis=0)
    df = df.loc[(fun(df["rawRed"]) > m[0] + 0.5 * s[0]) |
                (fun(df["rawRed"]) < m[0] - 0.5 * s[0]) |
                (fun(df["rawGreen"]) > m[1] + 0.5 * s[1]) |
                (fun(df["rawGreen"]) < m[1] - 0.5 * s[1])]
    return df


def aux_filter_rectangular2(df):
    y1_mean, y1_std = 354, 560
    y2_mean, y2_std = 222, 83
    df = df.loc[(df["rawRed"] > y1_mean + 0.3 * y1_std) |
                (df["rawRed"] < y1_mean - 0.3 * y1_std) |
                (df["rawGreen"] > y2_mean + 0.5 * y2_std) |
                (df["rawGreen"] < y2_mean - 0.5 * y2_std)]
    return df


class Iterator(object):
    def __init__(self, root_path, dir_list, crop_sz, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                 shuffle=True, infinite_loop=True, batch_sz=32, target_columns=("rawRed", "rawGreen")):
        self.lock = threading.Lock()
        self.root_path = root_path
        self.dir_list = dir_list
        self.crop_sz = crop_sz
        self.dir_tmpl = dir_tmpl
        self.shuffle = shuffle
        self.infinite_loop = infinite_loop
        self.batch_sz = batch_sz
        self.target_columns = list(target_columns)
        self.aux_filter = aux_filter
        self.batch_iterator = self.next()

    def next(self):
        batch_x = np.zeros((self.batch_sz, 3, self.crop_sz, self.crop_sz), dtype=K.floatx())
        batch_y = np.zeros((self.batch_sz, 2), dtype=K.floatx())
        i = 0
        keep_iterate = True
        while keep_iterate:
            keep_iterate = self.infinite_loop
            if self.shuffle:
                np.random.shuffle(self.dir_list)
            for d in self.dir_list:
                sd = self.root_path / self.dir_tmpl.format(d)

                channels = []
                for ch in [0, 3, 4]:
                    channels.append(cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U))
                channels = np.stack(channels)
                h, w = channels.shape[1:]

                df = pd.read_csv(sd / "Cells_rawints.csv")[["ObjectNumber",
                                                            "Location_Center_X",
                                                            "Location_Center_Y"] + self.target_columns]
                df = df[~df.isnull().any(axis=1)]
                if self.aux_filter: df = self.aux_filter(df)

                df["left"] = df["Location_Center_X"].astype(int) - self.crop_sz // 2
                df["top"] = df["Location_Center_Y"].astype(int) - self.crop_sz // 2
                df["right"] = df["left"] + self.crop_sz
                df["bottom"] = df["top"] + self.crop_sz
                df = df[(df["top"] >= 0) & (df["left"] >= 0) & (df["bottom"] <= h) & (df["right"] <= w)]
                if self.shuffle:
                    df = df.sample(frac=1)

                for _, row in df.iterrows():
                    top, bottom, left, right = row[["top", "bottom", "left", "right"]].astype(int).values
                    batch_x[i] = channels[:, top: bottom, left: right]
                    batch_y[i] = row[self.target_columns].values
                    i += 1
                    if i == self.batch_sz:
                        i = 0
                        yield preprocess_x(batch_x), preprocess_y(batch_y)
        if i > 0:
            batch_x, batch_y = batch_x[:i], batch_y[:i]
            yield preprocess_x(batch_x), preprocess_y(batch_y)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        with self.lock:
            batch_x, batch_y = next(self.batch_iterator)
        return batch_x, batch_y


class Iterator2(object):
    def __init__(self, root_path, dir_list, crop_sz, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                 shuffle=True, infinite_loop=True, batch_sz=32, target_columns=("rawRed", "rawGreen")):
        self.lock = threading.Lock()
        self.root_path = root_path
        self.dir_list = dir_list
        self.crop_sz = crop_sz
        self.dir_tmpl = dir_tmpl
        self.shuffle = shuffle
        self.infinite_loop = infinite_loop
        self.batch_sz = batch_sz
        self.target_columns = list(target_columns)
        self.aux_filter = aux_filter
        self.batch_iterator = self.next()

    def next(self):
        batch_x = np.zeros((self.batch_sz, 3, self.crop_sz, self.crop_sz), dtype=K.floatx())
        batch_y = np.zeros((self.batch_sz, 2), dtype=K.floatx())
        i = 0
        keep_iterate = True
        dir_images = {d: None for d in self.dir_list}
        dir_dfs = {d: None for d in self.dir_list}
        while keep_iterate:
            keep_iterate = self.infinite_loop
            if self.shuffle:
                np.random.shuffle(self.dir_list)
            for d in self.dir_list:
                if dir_images[d] is None:
                    sd = self.root_path / self.dir_tmpl.format(d)

                    dir_images[d] = np.stack(cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U)
                                             for ch in [0, 3, 4])
                    h, w = dir_images[d].shape[1:]

                    df = pd.read_csv(sd / "Cells_rawints.csv")[["ObjectNumber",
                                                                "Location_Center_X",
                                                                "Location_Center_Y"] + self.target_columns]
                    df = df[~df.isnull().any(axis=1)]
                    if self.aux_filter: df = self.aux_filter(df)

                    df["left"] = df["Location_Center_X"].astype(int) - self.crop_sz // 2
                    df["top"] = df["Location_Center_Y"].astype(int) - self.crop_sz // 2
                    df["right"] = df["left"] + self.crop_sz
                    df["bottom"] = df["top"] + self.crop_sz
                    df = df[(df["top"] >= 0) & (df["left"] >= 0) & (df["bottom"] <= h) & (df["right"] <= w)]
                    dir_dfs[d] = df

                if self.shuffle:
                    rows = dir_dfs[d].sample(n=1)
                else:
                    rows = dir_dfs[d]

                for _, row in rows.iterrows():
                    top, bottom, left, right = row[["top", "bottom", "left", "right"]].astype(int).values
                    batch_x[i] = dir_images[d][:, top: bottom, left: right]
                    batch_y[i] = row[self.target_columns].values
                    i += 1
                    if i == self.batch_sz:
                        i = 0
                        yield preprocess_x(batch_x), preprocess_y(batch_y)
        if i > 0:
            batch_x, batch_y = batch_x[:i], batch_y[:i]
            yield preprocess_x(batch_x), preprocess_y(batch_y)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        with self.lock:
            batch_x, batch_y = next(self.batch_iterator)
        return batch_x, batch_y


def remap_class(y):
    class_map = {
        0:  1,
        1:  1,
        2:  1,
        3:  4,
        4:  0,
        5:  2,
        6:  2,
        7:  4,
        8:  0,
        9:  3,
        10: 3,
        11: 4,
        12: 0,
        13: 0,
        14: 5,
        15: 5,
    }
    assert len(set(class_map.values())) == n_classes
    return class_map[y]


def remap_class4(y):
    class_map = {
        0:  0,
        1:  0,
        2:  1,
        3:  1,
        4:  0,
        5:  0,
        6:  1,
        7:  1,
        8:  2,
        9:  2,
        10: 3,
        11: 3,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
    }
    assert len(set(class_map.values())) == n_classes
    return class_map[y]


class Iterator_cls(object):
    def __init__(self, root_path, dir_list, crop_sz, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                 shuffle=True, infinite_loop=True, batch_sz=32, n_classes=4, target_column="class_4x4",
                 intensity_cols=(), output_intensities=False, remap_class=lambda cls: cls):
        self.lock = threading.Lock()
        self.root_path = root_path
        self.dir_list = dir_list
        self.crop_sz = crop_sz
        self.dir_tmpl = dir_tmpl
        self.shuffle = shuffle
        self.infinite_loop = infinite_loop
        self.batch_sz = batch_sz
        self.target_column = target_column
        self.n_classes = n_classes
        self.aux_filter = aux_filter
        self.output_intensities = output_intensities
        self.intensity_cols = list(intensity_cols)
        self.remap_class = remap_class
        self.batch_iterator = self.next()

    def next(self):
        batch_x = np.zeros((self.batch_sz, n_channels, self.crop_sz, self.crop_sz), dtype=K.floatx())
        batch_y = np.zeros((self.batch_sz, self.n_classes), dtype=K.floatx())
        batch_intensities = np.zeros((self.batch_sz, 2), dtype=K.floatx())
        i = 0
        keep_iterate = True
        dir_images = {d: None for d in self.dir_list}
        dir_dfs = {d: None for d in self.dir_list}
        while keep_iterate:
            keep_iterate = self.infinite_loop
            if self.shuffle:
                np.random.shuffle(self.dir_list)
            for d in self.dir_list:
                if dir_images[d] is None:
                    sd = self.root_path / self.dir_tmpl.format(d)

                    dir_images[d] = np.stack([cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U)
                                              for ch in [0, 3, 4]])
                    h, w = dir_images[d].shape[1:]

                    df = pd.read_csv(sd / "Cells_rawints20.csv")[["ObjectNumber",
                                                                  "Location_Center_X",
                                                                  "Location_Center_Y"] +
                                                                 [self.target_column] + self.intensity_cols]
                    df = df[~df.isnull().any(axis=1)]
                    if self.aux_filter: df = self.aux_filter(df)

                    df["left"] = df["Location_Center_X"].astype(int) - self.crop_sz // 2
                    df["top"] = df["Location_Center_Y"].astype(int) - self.crop_sz // 2
                    df["right"] = df["left"] + self.crop_sz
                    df["bottom"] = df["top"] + self.crop_sz
                    df = df[(df["top"] >= 0) & (df["left"] >= 0) & (df["bottom"] <= h) & (df["right"] <= w)]
                    dir_dfs[d] = df

                if self.shuffle:
                    rows = dir_dfs[d].sample(n=1)
                else:
                    rows = dir_dfs[d]

                for _, row in rows.iterrows():
                    top, bottom, left, right = row[["top", "bottom", "left", "right"]].astype(int).values
                    batch_x[i] = dir_images[d][:, top: bottom, left: right]
                    batch_y[i] = to_categorical(self.remap_class(row[self.target_column]), num_classes=self.n_classes)
                    if self.output_intensities:
                        batch_intensities[i] = row[self.intensity_cols].values
                    i += 1
                    if i == self.batch_sz:
                        i = 0
                        if self.output_intensities:
                            yield preprocess_x(batch_x), batch_y.copy(), batch_intensities.copy()
                        else:
                            yield preprocess_x(batch_x), batch_y.copy()
        if i > 0:
            batch_x, batch_y, batch_intensities = batch_x[:i], batch_y[:i], batch_intensities[:i]
            if self.output_intensities:
                yield preprocess_x(batch_x), batch_y.copy(), batch_intensities.copy()
            else:
                yield preprocess_x(batch_x), batch_y.copy()

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        with self.lock:
            outputs = next(self.batch_iterator)
        return outputs


class Iterator_Masked_cls(object):
    def __init__(self, root_path, dir_list, crop_sz, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                 shuffle=True, infinite_loop=True, batch_sz=32, n_classes=4, target_column="class_4x4",
                 intensity_cols=(), output_intensities=False, remap_class=lambda cls: cls):
        self.lock = threading.Lock()
        self.root_path = root_path
        self.dir_list = dir_list
        self.crop_sz = crop_sz
        self.dir_tmpl = dir_tmpl
        self.shuffle = shuffle
        self.infinite_loop = infinite_loop
        self.batch_sz = batch_sz
        self.target_column = target_column
        self.n_classes = n_classes
        self.aux_filter = aux_filter
        self.output_intensities = output_intensities
        self.intensity_cols = list(intensity_cols)
        self.remap_class = remap_class
        self.batch_iterator = self.next()

        self.dir_images = {d: None for d in self.dir_list}
        self.dir_dfs = {d: None for d in self.dir_list}
        for d in self.dir_list:
            sd = self.root_path / self.dir_tmpl.format(d)

            self.dir_images[d] = np.stack(cv2.imread(str(sd / "chan_{}.tiff".format(ch)), cv2.CV_16U)
                                     for ch in [0, 3, 4])

            if n_channels > 3:
                mask = cv2.imread(str(sd / "Labelled_cells.tiff"), cv2.CV_16U)
                mask[mask > 0] = 1
                self.dir_images[d] = np.concatenate((self.dir_images[d], mask[None, ...]))

            h, w = self.dir_images[d].shape[1:]

            df = pd.read_csv(sd / "Cells_rawints.csv")[["ObjectNumber",
                                                        "Location_Center_X",
                                                        "Location_Center_Y"] +
                                                       [self.target_column] + self.intensity_cols]
            df = df[~df.isnull().any(axis=1)]
            if self.aux_filter: df = self.aux_filter(df)

            df["left"] = df["Location_Center_X"].astype(int) - self.crop_sz // 2
            df["top"] = df["Location_Center_Y"].astype(int) - self.crop_sz // 2
            df["right"] = df["left"] + self.crop_sz
            df["bottom"] = df["top"] + self.crop_sz
            df = df[(df["top"] >= 0) & (df["left"] >= 0) & (df["bottom"] <= h) & (df["right"] <= w)]
            self.dir_dfs[d] = df

    def next(self):
        batch_x = np.zeros((self.batch_sz, n_channels, self.crop_sz, self.crop_sz), dtype=K.floatx())
        batch_y = np.zeros((self.batch_sz, self.n_classes), dtype=K.floatx())
        batch_intensities = np.zeros((self.batch_sz, 2), dtype=K.floatx())
        i = 0
        keep_iterate = True
        while keep_iterate:
            keep_iterate = self.infinite_loop
            if self.shuffle:
                np.random.shuffle(self.dir_list)
            for d in self.dir_list:
                if self.shuffle:
                    rows = self.dir_dfs[d].sample(n=1)
                else:
                    rows = self.dir_dfs[d]

                for _, row in rows.iterrows():
                    top, bottom, left, right = row[["top", "bottom", "left", "right"]].astype(int).values
                    batch_x[i] = self.dir_images[d][:, top: bottom, left: right]
                    batch_y[i] = to_categorical(self.remap_class(row[self.target_column]), num_classes=self.n_classes)
                    if self.output_intensities:
                        batch_intensities[i] = row[self.intensity_cols].values
                    i += 1
                    if i == self.batch_sz:
                        i = 0
                        if self.output_intensities:
                            yield preprocess_x(batch_x), batch_y.copy(), batch_intensities.copy()
                        else:
                            yield preprocess_x(batch_x), batch_y.copy()
        if i > 0:
            batch_x, batch_y, batch_intensities = batch_x[:i], batch_y[:i], batch_intensities[:i]
            if self.output_intensities:
                yield preprocess_x(batch_x), batch_y.copy(), batch_intensities.copy()
            else:
                yield preprocess_x(batch_x), batch_y.copy()

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        with self.lock:
            outputs = next(self.batch_iterator)
        return outputs


def add_classes(csv_name="Cells_rawints.csv", n_Red=4, n_Green=4):
    r = range(1, 131)
    dfs = []
    for i in r:
        df_name = root / "Seq0000_T1_XY{:03d}".format(i) / csv_name
        df = pd.read_csv(df_name)
        df["i"] = i
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs["clsRed"] = pd.qcut(dfs["rawRed"], n_Red).values.codes

    dfs["clsGreen"] = -1
    for cls in range(n_Red):
        dfs.loc[dfs["clsRed"] == cls, "clsGreen"] = \
            pd.qcut(dfs.loc[dfs["clsRed"] == cls, "rawGreen"], n_Green).values.codes
    dfs["class_{}x{}".format(n_Red, n_Green)] = (dfs["clsRed"] + dfs["clsGreen"] * n_Red).astype(int)
    dfs.loc[(dfs["clsRed"] == -1) | (dfs["clsGreen"] == -1), "class_{}x{}".format(n_Red, n_Green)] = np.nan
    del dfs["clsRed"], dfs["clsGreen"]
    for i in r:
        df_name = root / "Seq0000_T1_XY{:03d}".format(i) / csv_name
        df = dfs[dfs["i"] == i]
        del df["i"]
        df.to_csv(df_name, index=False, na_rep="nan")


if __name__ == "__main__":

# Regression framework
#
    # model = m46r(include_top=True, lr=lr, loss_balance=loss_balance)
    #
    # train_gen = Iterator2(root, train_list, SZ, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
    #                       batch_sz=32, shuffle=True, infinite_loop=True, target_columns=target_columns)
    #
    # val_gen = Iterator(root, val_list, SZ, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
    #                    batch_sz=32, shuffle=False, infinite_loop=False, target_columns=target_columns)
    # x, y = zip(*val_gen)
    # y0, y1 = zip(*y)
    # x = np.concatenate(x)
    # y0 = np.concatenate(y0)
    # y1 = np.concatenate(y1)
    # validation_data = x, [y0, y1]
    #
    # model.fit_generator(
    #     train_gen,
    #     steps_per_epoch=len(train_list) * 1500 // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_data,
    #     workers=4
    # )
    # model.save_weights(weights_path)


# predict
#     model = m46r(include_top=False, weights_path=weights_path)
#     descriptors = model.predict(x)
#     with open(descriptors_path, "wb") as f:
#         pickle.dump((descriptors, np.stack([y0, y1]).T), f)


# Classification framework
#
    model = m46c(include_top=True, lr=lr, classes=n_classes, weights=init_weights)

    train_gen = Iterator_cls(root, train_list, SZ, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                             batch_sz=batch_size, shuffle=True, infinite_loop=True,
                             n_classes=n_classes, target_column="class_2x2", remap_class=lambda cls: cls)

    val_gen = Iterator_cls(root, val_list, SZ, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                           batch_sz=batch_size, shuffle=False, infinite_loop=False,
                           n_classes=n_classes, target_column="class_2x2", remap_class=lambda cls: cls)
    x, y = zip(*val_gen)
    x = np.concatenate(x)
    y = np.concatenate(y)
    validation_data = x, y

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_list) * 1500 // batch_size,
        epochs=epochs,
        validation_data=validation_data,
        workers=3
    )
    model.save_weights(weights_path)

# predict
    model = m46c(include_top=True, lr=lr, classes=n_classes, weights=weights_path)

    val_gen = Iterator_cls(root, val_list, SZ, dir_tmpl="Seq0000_T1_XY{:03d}", aux_filter=None,
                           batch_sz=batch_size, shuffle=False, infinite_loop=False,
                           n_classes=n_classes, target_column="class_2x2",
                           output_intensities=True, intensity_cols=["rawRed20", "rawGreen20"],
                           remap_class=lambda cls: cls)
    x, y, intensities = zip(*val_gen)
    x = np.concatenate(x)
    y = np.argmax(np.concatenate(y), 1)
    intensities = np.concatenate(intensities)

    descriptors = model.predict(x)
    with open(descriptors_path, "wb") as f:
        pickle.dump((y, descriptors, intensities), f)
