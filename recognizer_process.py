'''
  examine the various features and selectors to choose the best features/selector pair
'''

import numpy as np
import pandas as pd
from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC

from my_recognizer import recognize
from asl_utils import show_errors

from asl_data import AslDb


# initializes the database
asl = AslDb()

# features_ground
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']


# features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x'])) / asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y'])) / asl.df['speaker'].map(df_std['right-y'])
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x'])) / asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y'])) / asl.df['speaker'].map(df_std['left-y'])
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

# features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx'].pow(2) + asl.df['grnd-ry'].pow(2))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx'].pow(2) + asl.df['grnd-ly'].pow(2))
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

# features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'
asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


# features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
min_rx = asl.df['right-x'].min()
min_ry = asl.df['right-y'].min()
min_lx = asl.df['left-x'].min()
min_ly = asl.df['left-y'].min()

max_rx = asl.df['right-x'].max()
max_ry = asl.df['right-y'].max()
max_lx = asl.df['left-x'].max()
max_ly = asl.df['left-y'].max()

asl.df['rescaling-rx'] = (asl.df['right-x'] - min_rx)/ (max_rx - min_rx)
asl.df['rescaling-ry'] = (asl.df['right-y'] - min_ry)/ (max_ry - min_ry)
asl.df['rescaling-lx'] = (asl.df['left-x'] - min_lx)/ (max_lx - min_lx)
asl.df['rescaling-ly'] = (asl.df['left-y'] - min_ly)/ (max_ly - min_ly)

features_rescaling = ['rescaling-rx', 'rescaling-ry', 'rescaling-lx', 'rescaling-ly']


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict


# features_ground with SelectorCV
features_set ={"features_ground": features_ground,
                "features_norm": features_norm,
                "features_polar": features_polar,
                "features_delta": features_delta,
                "features_rescaling": features_rescaling}
selector_set = {"SelectorCV": SelectorCV, "SelectorBIC": SelectorBIC, "SelectorDIC": SelectorDIC}

for features_key in features_set:
    features = features_set[features_key]
    for selector_key in selector_set:
        model_selector = selector_set[selector_key]
        print("\n------------------------------------")
        print("Processsing {} with {} selector".format(features_key, selector_key))
        print()
        models = train_all_words(features, model_selector)
        test_set = asl.build_test(features)
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)