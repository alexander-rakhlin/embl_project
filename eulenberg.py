
from pathlib import Path
from os.path import basename
import cv2
import numpy as np
import pickle

import umap
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(0)
kernel_initializer = "he_normal"

root = Path("data/CellCycle")

stages = ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"]
# stages = ["Anaphase"]


def vgg_block(num_filters, block_num, sz=3):
    def f(input_):
        if K.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        b = "block" + str(block_num)
        x = Convolution2D(num_filters, (sz, sz), kernel_initializer=kernel_initializer, padding="same",
                          name=b + "_conv1")(input_)
        x = ELU(name=b + "_elu1")(x)
        x = BatchNormalization(axis=bn_axis, name=b + "_bn1")(x)
        x = Convolution2D(num_filters, (1, 1), kernel_initializer=kernel_initializer, name=b + "_conv2")(x)
        x = ELU(name=b + "_elu2")(x)
        x = BatchNormalization(axis=bn_axis, name=b + "_bn2")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name=b + "_pool")(x)
        return x

    return f


def m46(include_top=True, input_shape=None, lr=1e-3, weights_path=None, classes=1000):

    if input_shape is None:
        if K.image_data_format() == "channels_last":
            input_shape = (None, None, 3)
        else:
            input_shape = (3, None, None)
    img_input = Input(shape=input_shape)

    # Convolution blocks
    x = vgg_block(32, 1)(img_input)  # Block 1
    x = vgg_block(64, 2)(x)  # Block 2
    x = vgg_block(128, 3)(x)  # Block 3
    x = vgg_block(256, 4)(x)  # Block 4
    x = vgg_block(512, 5)(x)  # Block 5

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    if include_top:
        # Classification block
        x = Dropout(0.3, name="dropout1")(x)
        x = Dense(1024, kernel_initializer=kernel_initializer, name="fc1")(x)
        x = ELU()(x)
        x = Dropout(0.3, name="dropout2")(x)
        x = ELU()(x)
        x = Dense(classes, activation="softmax", kernel_initializer=kernel_initializer, name="predictions")(x)

    # Create model.
    model = Model(img_input, x, name="m46")

    # load weights
    if weights_path is not None:
        print("Load weights from", weights_path)
        model.load_weights(weights_path)

    optimizer = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def encode(img, model):
    x = (img.astype(K.floatx()) / 255.0) * 2.0 - 1.0
    if K.image_data_format() == "channels_first":
        x = np.moveaxis(x, -1, 0)
    x = x[None, ...]
    dscr = model.predict(x)
    return dscr[0]


def extract_descriptors():
    from keras.models import Model
    from keras.layers import Dense, Input, Dropout
    from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
    from keras import backend as K
    from keras.optimizers import Adam
    from keras.layers.advanced_activations import ELU
    from keras.layers.normalization import BatchNormalization

    model = m46(include_top=False)
    data = {}
    for label, stage in enumerate(stages):
        print(stage)

        ids = set(basename(f).split("_")[0] for f in (root / stage).glob("*.jpg"))
        for id in ids:
            img_ch3 = cv2.imread(str(root / stage / "{}_Ch3.ome.jpg".format(id)), cv2.IMREAD_GRAYSCALE)
            img_ch4 = cv2.imread(str(root / stage / "{}_Ch4.ome.jpg".format(id)), cv2.IMREAD_GRAYSCALE)
            img_ch6 = cv2.imread(str(root / stage / "{}_Ch6.ome.jpg".format(id)), cv2.IMREAD_GRAYSCALE)
            img = np.stack([img_ch3, img_ch4, img_ch6], axis=-1)

            descriptor = encode(img, model)
            data[id] = {"descriptor": descriptor, "label": label}
    with open("data/descriptors.pkl", "wb") as f:
        pickle.dump(data, f)


def draw_umap(data, labels, hovertext=None, n_neighbors=15, min_dist=0.1,
              n_components=2, metric='euclidean', title='',  interactive3d=False):
    plt.style.use('_classic_test')
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)

    if n_components == 3 and interactive3d:
        import plotly
        import plotly.graph_objs as go
        plotly.offline.init_notebook_mode(connected=True)
        x, y, z = u.T
        trace1 = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                symbol="circle",
                size=3,
                color=labels,  # set color to an array/list of desired values
                colorscale='Rainbow',  # choose a colorscale
                opacity=1,
            ),
            hovertext=hovertext,
            hoverinfo="text",
        )
        dt = [trace1]
        layout = go.Layout(
            title=title
        )
        fig = go.Figure(data=dt, layout=layout)
        plotly.offline.iplot(fig)
    else:
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], range(len(u)), c=labels)
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], u[:, 1], c=labels)
        if n_components == 3 and not interactive3d:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels)
        plt.title(title, fontsize=18)
        plt.show()


if __name__ == "__main__":
    # extract_descriptors()
    with open("data/descriptors.pkl", "rb") as f:
        data = pickle.load(f)
    hovertext, x, labels = list(zip(*[(k, v["descriptor"], v["label"]) for k, v in data.items()]))
    x = np.stack(x)
    draw_umap(x, labels, hovertext, n_components=3, title="", interactive3d=False)
