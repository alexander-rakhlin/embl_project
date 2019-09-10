from datagen import Iterator
from models import m46c, r34, xcpt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from keras.callbacks import ModelCheckpoint, CSVLogger
from utils import double_division_tracks, curated_tracks


ROOT = Path('data/Timelapse_2019')
INTENSITY_COLS = ['GFP_20', 'Cy3_20']
CHANNEL_ROOTS = [ROOT / d for d in ['DAPI', 'BF']]
TARGET_COLUMN = 'sq20_cls2x2'
CROP_SZ = 48
LR = 1e-3

MODEL = r34
CLASSES = 4
CHANNELS = 2
BATCH_SIZE = 32
EPOCHS = 20
VERBOSE = False
INIT_WEIGHTS = None  # 'checkpoints/checkpoint.r34.sz48.03-0.68.hdf5'
MODEL_CHECKPOINT = f'checkpoints/checkpoint.{MODEL.__name__}.sz{CROP_SZ}.{{epoch:02d}}-{{val_acc:.2f}}.hdf5'
CSV_LOGGER = CSVLogger(f'logs/{MODEL.__name__}.sz{CROP_SZ}.log', append=True)

FRAMES = range(0, 200, 1)
VAL_TRACKS = list(double_division_tracks)
TRAIN_TRACKS = [t for t in curated_tracks if t not in VAL_TRACKS]
CELL_DF = pd.read_csv(ROOT / 'statistics_mean_std.csv')
DESCRIPTORS_PATH = ROOT / f'descriptors_all.{MODEL.__name__}.sz{CROP_SZ}.pkl'


def filter_cells(cell_df, frames='all', tracks='all'):
    if frames != 'all':
        cell_df = cell_df.loc[cell_df['FRAME'].isin(frames)]
    if tracks != 'all':
        cell_df = cell_df.loc[cell_df['TRACK_ID'].isin(tracks)]
    return cell_df


def train_test_split(cell_df, frames, train_tracks, test_tracks):
    train_df = cell_df.loc[cell_df['FRAME'].isin(frames) & cell_df['TRACK_ID'].isin(train_tracks)]
    test_df = cell_df.loc[cell_df['FRAME'].isin(frames) & cell_df['TRACK_ID'].isin(test_tracks)]
    return train_df, test_df


def train():
    train_df, val_df = train_test_split(CELL_DF, FRAMES, TRAIN_TRACKS, VAL_TRACKS)
    model = MODEL(channels=CHANNELS, lr=LR, include_top=True, classes=CLASSES, weights=INIT_WEIGHTS)
    train_iterator = Iterator(CHANNEL_ROOTS, train_df, CROP_SZ,
                              shuffle=True, seed=None, infinite_loop=True, batch_size=BATCH_SIZE,
                              classes=CLASSES, target_column=TARGET_COLUMN,
                              intensity_cols=INTENSITY_COLS, output_intensities=False,
                              output_df_index=False, verbose=VERBOSE, gen_id='train')
    val_iterator = Iterator(CHANNEL_ROOTS, val_df, CROP_SZ,
                            shuffle=False, seed=None, infinite_loop=False, batch_size=BATCH_SIZE,
                            classes=CLASSES, target_column=TARGET_COLUMN,
                            intensity_cols=INTENSITY_COLS, output_intensities=False,
                            output_df_index=False, verbose=VERBOSE, gen_id='val')
    x, y = zip(*val_iterator)
    x = np.concatenate(x)
    y = np.concatenate(y)
    validation_data = x, y
    callbacks = [ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_acc', save_best_only=True),
                 CSV_LOGGER]
    model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_data,
        validation_steps=len(val_df) // BATCH_SIZE,
        workers=3,
        callbacks=callbacks
    )


def predict(weights_path, df, descriptors_path):
    model = MODEL(channels=CHANNELS, include_top=True, classes=CLASSES, weights=weights_path)

    val_iterator = Iterator(CHANNEL_ROOTS, df, CROP_SZ,
                            shuffle=False, seed=None, infinite_loop=False, batch_size=BATCH_SIZE,
                            classes=CLASSES, target_column=TARGET_COLUMN,
                            intensity_cols=INTENSITY_COLS, output_intensities=True,
                            output_df_index=True, verbose=True, gen_id='val')

    x, y, intensities, df_index = zip(*val_iterator)
    x = np.concatenate(x)
    y = np.concatenate(y)
    intensities = np.concatenate(intensities)
    df_index = np.concatenate(df_index)

    descriptors = model.predict(x, batch_size=BATCH_SIZE)
    with open(descriptors_path, 'wb') as f:
        pickle.dump((y, descriptors, intensities, df_index), f)


def predict_batched(weights_path, df, descriptors_path):
    model = MODEL(channels=CHANNELS, include_top=True, classes=CLASSES, weights=weights_path)

    val_iterator = Iterator(CHANNEL_ROOTS, df, CROP_SZ,
                            shuffle=False, seed=None, infinite_loop=False, batch_size=BATCH_SIZE,
                            classes=CLASSES, target_column=TARGET_COLUMN,
                            intensity_cols=INTENSITY_COLS, output_intensities=True,
                            output_df_index=True, verbose=True, gen_id='val')
    y = []
    df_index = []
    intensities = []
    descriptors = []
    for x_, y_, intensities_, df_index_ in val_iterator:
        descriptors_ = model.predict_on_batch(x_)
        y.extend(y_)
        df_index.extend(df_index_)
        intensities.extend(intensities_)
        descriptors.extend(descriptors_)

    y = np.array(y)
    df_index = np.array(df_index)
    intensities = np.array(intensities)
    descriptors = np.array(descriptors)

    with open(descriptors_path, 'wb') as f:
        pickle.dump((y, descriptors, intensities, df_index), f)


if __name__ == '__main__':
    # train()

    # _, pred_df = train_test_split(CELL_DF, frames=range(200), train_tracks=TRAIN_TRACKS, test_tracks=VAL_TRACKS)
    pred_df = filter_cells(CELL_DF, frames='all')
    predict_batched('checkpoints/checkpoint.r34.sz48.03-0.73.hdf5', pred_df, DESCRIPTORS_PATH)
