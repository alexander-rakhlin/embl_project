import numpy as np
import umap
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
from progressbar import progressbar

def binarize(arr, threshold = 0.5):
    res = np.asarray(arr).copy()
    res[res < threshold] = 0
    res[res > 0] = 1
    return res.astype(int)

def binarize_encoding(encoding, threshold = 0.5):
    return lambda img: encoding(binarize(img, threshold))

def projection_encode(dim = 100):
    def encode(img):
        h,w,c = img.shape
        chan_to_vec = np.random.rand(c, dim)
        wv_img = (img.reshape(h, w, c, -1) *
                  chan_to_vec.reshape(1,1,c, dim)).sum(axis=2)
        return wv_img
    return encode

def id_encode():
    def encode(img):
        return img
    return encode


def umap_encode(dim = 100, n_neighbors = 100, min_dist =0.9, metric ="manhattan"):
    def encode(img):
        h,w,c = img.shape
        fit = umap.UMAP(
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            n_components = dim,
            metric = metric)
        return fit.fit_transform(img.reshape(-1, c)).reshape(h,w, -1)
    return encode

def pca_encode(dim = 100):
    def encode(img):
        h,w,c = img.shape
        fit = PCA(n_components = dim)
        return fit.fit_transform(img.reshape(-1, c)).reshape(h,w, -1)
    return encode

import gensim
from gensim.models import KeyedVectors

def wv_encode(path, image):
    wv = KeyedVectors.load_word2vec_format(str(path))
    vec_size = wv.vector_size
    chan_to_vec = np.zeros((image.data.shape[2], vec_size), dtype = np.float32)
    for ch in range(image.data.shape[2]):
        str_ion = str(image.chan_to_id(ch))
        if str_ion in wv:
            chan_to_vec[ch] = wv.word_vec(str_ion)
    chan_to_vec = np.array(chan_to_vec)

    def encode(img):
        h,w,c = img.shape
        wv_img = (img.reshape(h, w, c, -1) *
                  chan_to_vec.reshape(1,1,c, vec_size)).sum(axis=2)
        return wv_img
    return encode

class ConstRegression:
    def fit(self, data, result):
        self.const = result[0]

    def predict(self, data):
        return np.full(data.shape[0], self.const)

    def predict_proba(self, data):
        if self.const == 0:
            ret = np.zeros((data.shape[0], 2))
            ret[::,0] = 1
            return ret
        else:
            ret = np.zeros((data.shape[0], 2))
            ret[::,1] = 0
            return ret

class LogisticRegression:
    def __init__(self):
        pass
    
    def fit(self, data, result):
        n, f = data.shape
        n_, c = result.shape
        assert(n == n_)
        class_by_n = np.transpose(result)
        self.regressions = []
        for cls in class_by_n:
            if (cls == cls[0]).all():
                log = ConstRegression()
            else:
                log = lm.LogisticRegression()
            log.fit(data, cls)
            self.regressions.append(log)

    def predict(self, data):
        n, f = data.shape
        class_by_n = []
        for log in self.regressions:
            class_by_n.append(log.predict_proba(data)[::,1])
        return np.transpose(class_by_n)


class EncodeRegImgReducer:
    def __init__(self, encode,
                 decode_reg_factory = LogisticRegression, bin_learn = True):
        self.encode = encode
        self.decode_reg_factory = decode_reg_factory
        self.bin_learn = bin_learn

    def fit_transform(self, img):
        height, width, channels = img.shape
        encoded_img = self.encode(img)
        encoded_channels = encoded_img.shape[2]
        self.reg = self.decode_reg_factory()
        self.reg.fit(encoded_img.reshape(-1, encoded_channels),
                     binarize(img).reshape(-1, channels))
        return encoded_img

    def inverse_transform(self, encoded_img):
        height, width, encoded_channels = encoded_img.shape
        return self.reg.predict(
            encoded_img.reshape(-1, encoded_channels)).reshape(height, width, -1)

class ArrayImgReducer:
    def __init__(self, reducer, bin_learn = True):
        self.reducer = reducer
        self.binarize = bin_learn

    def fit_transform(self, img):
        if (self.binarize):
            img = binarize(img)
        height, width, channels = img.shape
        return self.reducer.fit_transform(img.reshape(-1,
                                                      channels)).reshape(height,
                                                                         width, -1)

    def inverse_transform(self, encoded_img):
        height, width, encoded_channels = encoded_img.shape
        return self.reducer.inverse_transform(
            encoded_img.reshape(-1,
                                encoded_channels)).reshape(height,
                                                           width, -1)
def f1(acc, pre):
    return 2 / (1/acc + 1/ pre)

    
def calc_f1(img, pred):
    ps = pred.sum()
    if ps == 0:
        return 0
    tp = (img & pred).sum()
    acc = tp / img.sum()
    pre = tp / ps
    return f1(acc, pre)
    
class BinarizeReducer:
    def __init__(self, reducer, threshold = 0.5):
        self.reducer = reducer
        self.img_threshold = threshold

    def fit_transform(self, img):
        ret = self.reducer.fit_transform(img)
        pred = self.reducer.inverse_transform(ret)
        bin_img = binarize(img, self.img_threshold)
        best_thresh = 0
        best_f1 = -1
        for thresh in np.arange(0,1.001, 0.005):
            bin_pred = binarize(pred, thresh)
            tf1 = calc_f1(bin_img, bin_pred)
            if tf1 > best_f1:
                best_f1 = tf1
                best_thresh = thresh
        self.threshold = best_thresh
        return ret
        
    def inverse_transform(self, img):
        ret = self.reducer.inverse_transform(img)
        return binarize(ret, self.threshold)
