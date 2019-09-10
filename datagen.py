import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import threading
from keras import backend as K
from keras.utils import to_categorical


def preprocess(x, means, stds):
    x = np.moveaxis(x, 1, 0)
    x = np.array([(ch - m[:, None, None]) / s[:, None, None] for ch, m, s in zip(x, means, stds)])
    return np.moveaxis(x, 0, 1)

class Iterator(object):
    """Iterator for 2019 data"""

    def __init__(self, channel_roots, cell_df, crop_sz,
                 shuffle=True, seed=None, infinite_loop=True, batch_size=32,
                 classes=4, target_column='class_2x2', intensity_cols=None, output_intensities=False,
                 output_df_index=False, verbose=False, gen_id=""):
        self.n_channels = 2
        self.lock = threading.Lock()
        self.channel_roots = channel_roots
        self.frames = sorted(cell_df['FRAME'].unique())
        self.cell_df = cell_df
        self.crop_sz = crop_sz
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.gen_id = gen_id
        self.total_batches_seen = 0
        self.infinite_loop = infinite_loop
        self.batch_size = batch_size
        self.target_column = target_column
        self.classes = classes
        self.output_intensities = output_intensities
        self.output_df_index = output_df_index
        self.intensity_cols = list(intensity_cols)
        self.index_generator = self._flow_index()

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.frame_index = 0
        self.batch_index = 0
        while 1:
            if self.seed is None:
                random_seed = None
            else:
                random_seed = self.seed + self.total_batches_seen

            # next frame, read images
            if self.batch_index == 0:
                if self.frame_index >= len(self.frames):
                    if not self.infinite_loop:
                        break
                    self.frame_index = 0
                if self.frame_index == 0:
                    if self.verbose:
                        print(f'\n************** New epoch. Generator {self.gen_id} *******************')
                    if self.shuffle:
                        np.random.RandomState(random_seed).shuffle(self.frames)

                frame_num = self.frames[self.frame_index]
                if self.verbose:
                    print(f'************** Frame T{frame_num + 1:0>3}, index {self.frame_index}. '
                          f'Generator {self.gen_id} *******************')

                channels = [cv2.imread(str(root / f'T{frame_num + 1:0>3}'), cv2.CV_16U) for root in self.channel_roots]
                h, w = channels[0].shape
                df = self.cell_df.loc[
                    (self.cell_df['FRAME'] == frame_num) &
                    (self.cell_df['POSITION_Y'] - self.crop_sz // 2 >= 0) &
                    (self.cell_df['POSITION_X'] - self.crop_sz // 2 >= 0) &
                    (self.cell_df['POSITION_Y'] - self.crop_sz // 2 + self.crop_sz < h) &
                    (self.cell_df['POSITION_X'] - self.crop_sz // 2 + self.crop_sz < w)
                    ]
                frame_len = len(df)
                if self.shuffle:
                    df = df.sample(frame_len)

                self.frame_index += 1

            current_index = (self.batch_index * self.batch_size) % frame_len
            if frame_len > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = frame_len - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield df.iloc[current_index: current_index + current_batch_size],\
                  current_index, \
                  current_batch_size,\
                  channels

    def next(self):
        with self.lock:
            df_slice, current_index, current_batch_size, channels = next(self.index_generator)

        batch_x = np.zeros((current_batch_size, self.n_channels, self.crop_sz, self.crop_sz), dtype=K.floatx())
        batch_y = np.zeros((current_batch_size, self.classes), dtype=K.floatx())
        if self.output_intensities:
            batch_intensities = np.zeros((current_batch_size, len(self.intensity_cols)), dtype=K.floatx())
        if self.output_df_index:
            batch_df_index = np.zeros((current_batch_size, 1), dtype=int)

        for i, (df_index, row) in enumerate(df_slice.iterrows()):
            top, left = row[['POSITION_Y', 'POSITION_X']].astype(int).values - self.crop_sz // 2
            batch_x[i] = np.stack([channel[top: top + self.crop_sz, left: left + self.crop_sz]
                                   for channel in channels])
            batch_y[i] = to_categorical(row[self.target_column], num_classes=self.classes)
            if self.output_intensities:
                batch_intensities[i] = row[self.intensity_cols].values
            if self.output_df_index:
                batch_df_index[i] = df_index

        batch_means = [df_slice[root.name + '_average'].values for root in self.channel_roots]
        batch_stds = [df_slice[root.name + '_std'].values for root in self.channel_roots]
        batch_x = preprocess(batch_x, batch_means, batch_stds)

        if K.image_data_format() == 'channels_last':
            batch_x = np.moveaxis(batch_x, 1, -1)

        result = batch_x, batch_y
        if self.output_intensities:
            result += (batch_intensities,)
        if self.output_df_index:
            result += (batch_df_index,)
        return result

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


if __name__ == "__main__":
    from itertools import islice
    import pickle

    double_division_tracks = {
        46: [13, 165],
        85: [0, 184],
        448: [33, 185],
        1552: [13, 188],
        1735: [9, 197],
        1897: [18, 195],
        1920: [45, 187],
        2151: [25, 162],
        2340: [54, 187],
        2660: [6, 161],
        2920: [61, 185],
        2963: [6, 196],
        2974: [0, 182],
        3029: [1, 196],
        3890: [21, 161],
        3985: [9, 164],
        4137: [4, 185],
        4997: [6, 160],
        6240: [8, 163],
        6382: [6, 167],
        6427: [6, 191],
        #     6603: [22, 190],
        6784: [31, 194],
        6950: [0, 176],
        7143: [2, 184],
        7899: [28, 196],
        8117: [0, 175],
        10300: [4, 194],
        10455: [8, 188],
        14298: [8, 178],
        14561: [8, 186],
        15006: [9, 176],
        17953: [12, 194],
        18009: [13, 171],
        18328: [14, 169],
        18367: [14, 186],
        18369: [14, 190],
        18397: [14, 179],
        24515: [22, 173],
        25101: [26, 183],
        25295: [29, 168],
        25313: [28, 195],
        25843: [32, 178],
        26568: [35, 179],
        #     26675: [35, 175],
        26934: [37, 181],
        27161: [39, 182],
        #     27265: [42, 191],
        27472: [42, 175],
        27592: [43, 193],
    }
    double_division_tracks = list(double_division_tracks)

    ROOT = Path('data/Timelapse_2019')
    cell_df = pd.read_csv(ROOT / 'statistics_mean_std.csv')
    frames = range(5)
    tracks = [0, 10009, 10013]
    tracks = double_division_tracks
    cell_df = cell_df.loc[cell_df['FRAME'].isin(frames) & cell_df['TRACK_ID'].isin(tracks)]
    crop_sz = 48
    channel_roots = [ROOT / d for d in ['DAPI', 'BF']]

    batch_iterator = Iterator(channel_roots, cell_df, crop_sz,
                              shuffle=True, seed=None, infinite_loop=True, batch_size=5,
                              classes=4, target_column='sq20_cls2x2',
                              intensity_cols=['GFP_20', 'Cy3_20'], output_intensities=True,
                              output_df_index=True, verbose=True, gen_id='1')

    x, y, intensities, df_index = zip(*islice(batch_iterator, None))
    x = np.concatenate(x)
    y = np.concatenate(y)
    intensities = np.concatenate(intensities)
    df_index = np.concatenate(df_index)

    # with open(ROOT / 'gen_test.pkl', 'wb') as f:
    #     pickle.dump((x, y, intensities, df_index), f)
    pass
