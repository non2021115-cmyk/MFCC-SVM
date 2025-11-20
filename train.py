import os
import pickle
import numpy as np
import librosa
import scipy.io.wavfile as sci_wav
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from core_functions.mfcc_feature_extraction import extract_mfcc_feature

# -----------------------
# Data path definitions
# -----------------------
fs = 16000
ROOT_DIR_GOOD = '../data/train/good/'
ROOT_DIR_BAD = '../data/train/bad/'
ROOT_DIR_GOOD_TEST = '../data/test/good/'
ROOT_DIR_BAD_TEST = '../data/test/bad/'
REAL_DATA_DIR = '../data/real/'     # <-- 추가됨

# -----------------------
# Utility functions
# -----------------------
def read_wav_files(root_dir, wav_files):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(root_dir + f)[1] for f in wav_files]

def read_train_test_data():
    wav_files_good = os.listdir(ROOT_DIR_GOOD)
    wav_files_bad  = os.listdir(ROOT_DIR_BAD)
    wav_files_good_test = os.listdir(ROOT_DIR_GOOD_TEST)
    wav_files_bad_test  = os.listdir(ROOT_DIR_BAD_TEST)

    data_good_train = read_wav_files(ROOT_DIR_GOOD, wav_files_good)
    data_bad_train  = read_wav_files(ROOT_DIR_BAD, wav_files_bad)
    data_good_test  = read_wav_files(ROOT_DIR_GOOD_TEST, wav_files_good_test)
    data_bad_test   = read_wav_files(ROOT_DIR_BAD_TEST, wav_files_bad_test)

    ds_train = data_bad_train + data_good_train
    y_train = np.concatenate((np.ones(len(data_bad_train)), np.zeros(len(data_good_train))))

    ds_test = data_bad_test + data_good_test
    y_test = np.concatenate((np.ones(len(data_bad_test)), np.zeros(len(data_good_test))))

    return ds_train, y_train, ds_test, y_test

def pad_data(data_set, fix_length):
    if not isinstance(data_set, list):
        return librosa.util.fix_length(data_set, size=fix_length, mode='wrap')

    data_set_fix = np.zeros((len(data_set), fix_length))
    for i, d in enumerate(data_set):
        data_set_fix[i, :] = librosa.util.fix_length(d, size=fix_length, mode='wrap')
    return data_set_fix

def mfcc_extraction(data_set_fix_len, fs, n_fft, frame_size, frame_step):
    flag = True
    for i, d in enumerate(data_set_fix_len):
        mfcc = extract_mfcc_feature(d, fs, n_fft, frame_size, frame_step)
        if flag:
            mfcc_features = np.zeros((len(data_set_fix_len), mfcc.shape[0], mfcc.shape[1]))
            flag = False
        mfcc_features[i] = mfcc
    return mfcc_features

def preprocess_raw_data(data, fix_length, params):
    data_fix = pad_data(data, fix_length)
    mfcc_feat = mfcc_extraction(data_fix,
                                params['fs'],
                                params['n_fft'],
                                params['frame_size'],
                                params['frame_step'])
    return mfcc_feat


# -----------------------
# REAL data preprocessing
# -----------------------
def preprocess_real_folder(pca, fix_len, params):
    wav_files = [f for f in os.listdir(REAL_DATA_DIR) if f.endswith('.wav')]

    if len(wav_files) == 0:
        print("No real wav files found.")
        return None

    real_raw = []
    for wf in wav_files:
        _, d = sci_wav.read(os.path.join(REAL_DATA_DIR, wf))
        d = librosa.util.fix_length(d, size=fix_len, mode='wrap')
        real_raw.append(d)

    # MFCC
    real_mfcc = mfcc_extraction(real_raw,
                                params['fs'],
                                params['n_fft'],
                                params['frame_size'],
                                params['frame_step'])

    # flatten
    real_flat = np.array([f.ravel() for f in real_mfcc])

    # PCA transform
    real_pca = pca.transform(real_flat)

    # save
    np.save("real_test_pca.npy", real_pca)
    print(f"Saved real_test_pca.npy ({real_pca.shape})")

    return real_pca


# -----------------------
# TRAIN FUNCTION
# -----------------------
def train():

    # Load data
    data_train, y_train, data_test, y_test = read_train_test_data()

    # parameters
    n_fft = 512
    params = {
        'fs': fs,
        'n_fft': n_fft,
        'frame_size': n_fft / fs,
        'frame_step': int(n_fft / 3) / fs,
    }

    fix_len_s = 13.5
    fix_len = int(fix_len_s * fs)

    # MFCC for train/test
    X_train = preprocess_raw_data(data_train, fix_len, params)
    X_train = np.array([f.ravel() for f in X_train])
    X_test = preprocess_raw_data(data_test, fix_len, params)
    X_test = np.array([f.ravel() for f in X_test])

    # PCA
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # SVM training
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train_pca, y_train)

    # Save model & PCA
    pickle.dump(svm, open("svm_model.pkl", "wb"))
    pickle.dump(pca, open("pca.pkl", "wb"))
    print("Saved svm_model.pkl & pca.pkl")

    # --- Real data PCA transform & save ---
    preprocess_real_folder(pca, fix_len, params)

    return svm, pca, X_train_pca, y_train, X_test_pca, y_test


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    train()
