import pandas as pd
import re
import numpy as np
from progressbar import progressbar
import molar_mass_calculator as mmc

class Image:

    def __init__(self, df, db, fdr = 51, percentile = 99, fq = 0):
        # fdr filter
        df = df[df.fdr < fdr]
        # intensity normalization, flashspot removal
        filt_df = []       
        for ind, ion_rows in df.groupby('ion_ind'):
            cnt = ion_rows.ion_ind.count()
            if cnt >= fq:
                int_thresh = ion_rows.int.quantile(percentile / 100)
                ion_rows.loc[ion_rows['int'] > int_thresh, 'int'] = int_thresh
                ion_rows.loc[:, 'int'] /= int_thresh
                filt_df += [ion_rows.copy()]
                
        df = pd.concat(filt_df)
        df = df.reset_index(drop=True)

        img_ions = sorted(df["ion_ind"].unique())
        def add_mass(ion):
            return (mmc.molar_mass(db.id_to_formula(ion)), ion)
        with_mass = sorted(map(add_mass, img_ions))
        img_ions = map(lambda x: x[1], with_mass)
        self.id_to_chan_ = {ion_ind: i for i, ion_ind in enumerate(img_ions)}
        self.chan_to_id_ = {i: ion_ind for i, ion_ind in enumerate(img_ions)}

        img = np.zeros((int(df.y.max()) + 1, int(df.x.max()) + 1, len(self.id_to_chan_)), dtype=np.float32)
        for _, row in df[['x', 'y', 'ion_ind', 'int']].iterrows():
            x, y, ion_ind, intensity = row
            img[int(y)][int(x)][self.id_to_chan_[ion_ind]] = intensity
        
        self.data = img
        self.db = db

    def id_to_chan(self, ind):
        return self.id_to_chan_[ind]

    def chan_to_id(self, ch):
        return self.chan_to_id_[ch]

    def chan_to_ion(self, ch):
        return self.db.id_to_ion(self.chan_to_id(ch))

    def ion_to_chan(self, ion):
        return self.id_to_chan(self.db.ion_to_id(ion))

    def chan_to_formula(self, ch):
        return self.db.id_to_formula(self.chan_to_id(ch))

class DB:
    def __init__(self, path):
        self.path = path
        self.ion_df = pd.read_msgpack(self.path / 'ion_df.msgpack')

    def id_to_ion(self, ion_id):
        db = self.ion_df.loc[ion_id, ["formula", "adduct"]]
        return "%s%s" % (db.formula, db.adduct)

    def id_to_formula(self, ion_id):
        db = self.ion_df.loc[ion_id, ["formula"]]
        return db.formula

    def ion_to_id(self, ion):
        match = re.search('(.*)([+-].*)', ion)
        formula = match.group(1)
        adduct = match.group(2)
        return self.ion_df.loc[(self.ion_df["formula"] == formula) &
                               (self.ion_df["adduct"] == adduct)].index[0]

    def load_img(self, ds_ind, fdr = 51, percentile = 99, fq = 0):
        img_df = pd.read_msgpack(self.path / 'pixel_df_list' / str(ds_ind))
        return Image(img_df, self, fdr, percentile, fq)
