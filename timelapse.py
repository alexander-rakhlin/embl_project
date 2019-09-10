from pathlib import Path
import numpy as np
import pandas as pd
import cv2


def get_frame_average(img, yx, sz, fun=np.median):
    sz2 = sz // 2
    mask = np.zeros_like(img, dtype=bool)
    for y, x in yx:
        left, top = max(0, x - sz2), max(0, y - sz2)
        mask[top: y + sz2, left: x + sz2] = True
    return fun(img[mask])


root = Path('data/Timelapse_2019')
gfp_root = root / 'GFP'
cy3_root = root / 'Cy3'

SZ = 20
SZ2 = SZ // 2

curated_tracks = sorted(pd.read_csv(root / 'curated_tracks.csv', header=None).astype(int).values.flatten())
df = pd.read_csv(root / 'Spots in tracks statistics.csv', na_values='None', delimiter='\t').dropna()
df = df[df['TRACK_ID'].isin(curated_tracks)]

frame_names = [f.name for f in gfp_root.glob('*')]
frame_names = sorted(frame_names, key=lambda s: int(s.split('.')[0][1:]))

div_frames = dict.fromkeys(curated_tracks)
rows = []
for frame_name in frame_names:
    print('Frame', frame_name)
    frame_num = int(frame_name.split('.')[0][1:]) - 1

    row = []
    gfp = cv2.imread(str(gfp_root / frame_name), cv2.CV_16U)
    cy3 = cv2.imread(str(cy3_root / frame_name), cv2.CV_16U)
    # green_mean = gfp.mean()
    # red_mean = cy3.mean()
    dt = df.loc[df['FRAME'] == frame_num, ['TRACK_ID', 'POSITION_X', 'POSITION_Y']].astype(int)
    yx = dt[['POSITION_Y', 'POSITION_X']].values
    gfp_frame_average = get_frame_average(gfp, yx, SZ, fun=np.median)
    cy3_frame_average = get_frame_average(cy3, yx, SZ, fun=np.median)
    row.extend([frame_num, gfp_frame_average, cy3_frame_average])

    for track in curated_tracks:
        dxy = dt[dt['TRACK_ID'] == track]
        if (dxy.shape[0] > 1) and (div_frames[track] is None):  # div_frame is where 2 cells
            div_frames[track] = frame_num
        if dxy.shape[0] < 1:
            time = np.nan  # div_frame
            x, y = np.nan, np.nan
            green_median = np.nan
            red_median = np.nan
            green_mean = np.nan
            red_mean = np.nan
        else:
            time = frame_num
            x, y = dxy[['POSITION_X', 'POSITION_Y']].values[0]
            left, top = max(0, x - SZ2), max(0, y - SZ2)
            green_median = np.median(gfp[top: y + SZ2, left: x + SZ2])
            red_median = np.median(cy3[top: y + SZ2, left: x + SZ2])
            green_mean = np.mean(gfp[top: y + SZ2, left: x + SZ2])
            red_mean = np.mean(cy3[top: y + SZ2, left: x + SZ2])
        row.extend([time, x, y, green_median, red_median, green_mean, red_mean])
    rows.append(row)

div_frames = {k: 0 if v is None else v for k, v in div_frames.items()}
columns = [('frame_num',), ('gfp_frame_average',), ('cy3_frame_average',)]
columns_ = [[(track, 'time'), (track, 'x'), (track, 'y')] +
            [(track, color, fun)
             for fun in ('median', 'mean')
             for color in ('green', 'red')]
            for track in curated_tracks]
columns.extend(tt for t in columns_ for tt in t)
dfo = pd.DataFrame.from_records(rows, columns=pd.MultiIndex.from_tuples(columns))
for t in curated_tracks:
    dfo[(t, 'time')] -= div_frames[t]
dfo.to_csv(root / 'intensities.csv', index=False)
