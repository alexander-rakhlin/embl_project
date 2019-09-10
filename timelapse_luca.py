import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm, os

def scale(arr1):
    s1 = (arr1 - np.min(arr1)) / (np.max(arr1) - np.min(arr1))
    return s1

im_folder = 'E:/Experiments/20180910_Timelapse/'
csv_p = im_folder + 'All Spots statistics.csv'
curated_tracks = [str(tracks[0]) for tracks in pd.read_csv('E:/Experiments/20180910_Timelapse/curated_tracks.csv').as_matrix()]

df = pd.read_csv(csv_p)

check_all_tracks = 0
if check_all_tracks:
    split_track = []
    for track in tqdm.tqdm(np.unique(df['TRACK_ID'])):
        if track != 'None':
            arr = df[df['TRACK_ID'] == track]['FRAME']
            frames, nID = np.unique(arr, return_counts=True)
            division_frame = np.where(nID == 2)
            if len(division_frame[0]) > 0:
                if nID[division_frame[0][0]] == nID[-1]:
                    split_track = np.append(split_track, track)

    # for track in tqdm.tqdm(split_track):
    #     arr = df[df['TRACK_ID'] == track]['FRAME']
    #     frames, nID = np.unique(arr, return_counts=True)
    #     plt.plot(nID)


    # Track curation: for each track creates images of the detected cells in a folder, to be manually curated for cell division
    show_image = 1
    for track_id in split_track[334:]:
        x_min, x_max = np.int(np.min(df[df['TRACK_ID'] == track_id]['POSITION_X'].as_matrix())), \
                       np.int(np.max(df[df['TRACK_ID'] == track_id]['POSITION_X'].as_matrix()))
        y_min, y_max = np.int(np.min(df[df['TRACK_ID'] == track_id]['POSITION_Y'].as_matrix())), \
                       np.int(np.max(df[df['TRACK_ID'] == track_id]['POSITION_Y'].as_matrix()))
        if np.min([x_min, x_max, y_min, y_max]) > 10:
            for frame in tqdm.tqdm(df[df['TRACK_ID'] == track_id]['FRAME'].as_matrix()):

                x = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track_id][df['FRAME'] == frame]['POSITION_X'].as_matrix()])
                y = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track_id][df['FRAME'] == frame]['POSITION_Y'].as_matrix()])

                im_g = plt.imread(im_folder + '/stitched_GFP/T{}'.format(str(frame + 1).zfill(2)))
                im_r = plt.imread(im_folder + '/stitched_cy3/T{}'.format(str(frame + 1).zfill(2)))

                if show_image:
                    im_bf = plt.imread(im_folder + '/stitched_DAPI/T{}'.format(str(frame + 1).zfill(2)))[y_min - 10:y_max + 10,
                              x_min - 10:x_max + 10]
                    im_bf_ct = np.clip(im_bf, np.percentile(im_bf, 1), np.percentile(im_bf, 99))

                    fig = plt.figure()
                    ax = plt.subplot(111)
                    plt.imshow(im_bf_ct, cmap='gray')
                    plt.scatter(x-x_min+10, y-y_min+10, 100, [1, 0, 0])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.patch.set_facecolor((0, 0, 0))
                    plt.tight_layout()

                    if not os.path.exists(im_folder + 'tracking/{}'.format(track_id)):
                        os.mkdir(im_folder + 'tracking/{}'.format(track_id))

                    plt.savefig(im_folder + 'tracking/{}/{}.png'.format(track_id, frame))
                    plt.close('all')

#Plot fluoresence profile for of the cells by respect to their division.
plt.figure()
plt.xlabel('Time (hr)', fontsize=15)
plt.ylabel('Average Fluo Intensity', fontsize=15)

trend_g = []
trend_r = []
for frame in range(96):
    # print(frame)
    im_g = plt.imread(im_folder + '/stitched_GFP/T{}'.format(str(frame + 1).zfill(2)))
    im_r = plt.imread(im_folder + '/stitched_cy3/T{}'.format(str(frame + 1).zfill(2)))

    val_g = np.mean(im_g)
    val_r = np.mean(im_r)
    trend_g = np.append(trend_g, val_g)
    trend_r = np.append(trend_r, val_r)

    plt.scatter(frame*10/60, val_g, 20, 'g') #express data in terns of time (hr), 1 frame every 10 min
    plt.scatter(frame*10/60, val_r, 20, 'r')

# curated_tracks_p = 'E:/Experiments/20180910_Timelapse/positive doubles/'
# curated_tracks = os.listdir(curated_tracks_p)
# pd.DataFrame(data=curated_tracks).to_csv('E:/Experiments/20180910_Timelapse/curated_tracks.csv', index=False)

# For each track, collect the mean intensity around the cells from each frame in cy3 and GFP channels
# Attention, time consuming part, takes about 2.5hrs on 1 core 4.5Ghz i7 and images stored on SSD
# To speed up, loop over frames, then loop over the track and collect data for the cells of that track in that frame

xvalS = []
greenS = []
redS = []
for track in tqdm.tqdm(curated_tracks):
    arr = df[df['TRACK_ID'] == track]['FRAME']
    frames, nID = np.unique(arr, return_counts=True)
    div_frame = frames[np.where(nID == 2)[0][0]]
    for f in frames:
        X = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track][df['FRAME'] == f]['POSITION_X'].as_matrix()])
        Y = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track][df['FRAME'] == f]['POSITION_Y'].as_matrix()])
        im_g = plt.imread(im_folder + '/stitched_GFP/T{}'.format(str(f + 1).zfill(2)))
        im_r = plt.imread(im_folder + '/stitched_cy3/T{}'.format(str(f + 1).zfill(2)))

        xval = f - div_frame

        for x,y in zip(X,Y):
            green = np.median(im_g[y - 10: y + 10, x - 10: x + 10]) - trend_g[f]
            red = np.median(im_r[y - 10: y + 10, x - 10: x + 10]) - trend_r[f]
            xvalS = np.append(xvalS, xval)
            greenS = np.append(greenS, green)
            redS = np.append(redS, red)

xvalS_h = xvalS *10/60 #express in hours, 1 frame every 10 min

plt.figure()
plt.xlabel('Time (hr), relative to division event', fontsize=15)
plt.ylabel('Fluo. intensity', fontsize=15)
plt.axvline(x=0, linewidth=2, color='k', label='Division event')
plt.legend()

for xval, green, red in zip(xvalS_h, greenS, redS):
    plt.scatter(xval, green, 10, color='g', alpha=0.1)
    plt.scatter(xval, red, 10, color='r', alpha=0.1)

for x in xvalS_h:
    inds = np.where(xvalS_h == x)[0]
    green_mean = np.mean([greenS[i] for i in inds])
    red_mean = np.mean([redS[i] for i in inds])

    plt.scatter(x, green_mean, 40, color=[0,1,0])
    plt.scatter(x, red_mean, 40, color=[1,0,0])












