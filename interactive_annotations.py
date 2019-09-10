import pickle
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)
import cv2
from pathlib import Path
from os.path import basename

plt.style.use('_classic_test')

def read_image(m, stage):
    PAD = 2
    SZ = 66
    N = 3
    img_ch3 = cv2.imread(str(root / stage / "{}_Ch3.ome.jpg".format(m)))
    img_ch4 = cv2.imread(str(root / stage / "{}_Ch4.ome.jpg".format(m)))
    img_ch6 = cv2.imread(str(root / stage / "{}_Ch6.ome.jpg".format(m)))
    img = np.ones((SZ, SZ * N + 2 * PAD, 3), dtype=np.uint8) * 255
    img[:, 0: SZ] = img_ch3
    img[:, SZ + PAD: 2 * SZ + PAD] = img_ch4
    img[:, 2 * SZ + 2 * PAD: 3 * SZ + 2 * PAD] = img_ch6
    return img


root = Path("data/CellCycle")
stages = ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"]
d = {}
for stage in stages:
    ids = set(basename(f).split("_")[0] for f in (root / stage).glob("*.jpg"))
    d.update({i: stage for i in ids})

with open("data/sample_embeddings.pkl", "rb") as f:
    hovertext, u, labels = pickle.load(f)

legend = ["G1", "S", "G2", "M"]
norm = mpl.colors.Normalize(vmin=0, vmax=len(legend) - 1)
cmap = mpl.cm.get_cmap()
m = cm.ScalarMappable(norm=norm, cmap=cmap)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
lines = []
dlines = [[]] * len(set(labels))
for i, label in enumerate(sorted(set(labels))):
    uu = u[labels == label]
    dlines[i] = hovertext[labels == label]
    lines.append(ax.scatter(uu[:, 0], uu[:, 1], c=[m.to_rgba(label)] * len(uu)))
plt.title("metric = euclidean", fontsize=18)
plt.legend(legend, loc="lower left")

# create the annotations box
arr_img = read_image("1", "G1")
im = OffsetImage(arr_img, cmap="Greys", zoom=0.7)
ab = AnnotationBbox(im, (0, 0), xybox=(0.8, 0.9), xycoords="data",
                    boxcoords="axes fraction", pad=0.3, arrowprops=None)
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)


def hover(event):
    for i, line in enumerate(lines):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            m = dlines[i][line.contains(event)[1]["ind"][0]]
            stage = d[m]
            img = read_image(m, stage)
            im.set_data(img)

            # make annotation box visible
            ab.set_visible(True)
            fig.canvas.draw_idle()
            return

    # if the mouse is not over a scatter point
    ab.set_visible(False)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('motion_notify_event', hover)
plt.show()
