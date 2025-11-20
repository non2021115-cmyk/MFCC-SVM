import os
import pickle
import numpy as np
import scipy.io.wavfile as sci_wav
import librosa
from mfcc_feature_extraction import extract_mfcc_feature

# -------------------------
# Load saved PCA & SVM
# -------------------------
svm = pickle.load(open("svm_model.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

# -------------------------
# Preprocessing parameters
# -------------------------
fs = 16000
n_fft = 512
fix_len_s = 13.5
fix_len = int(fix_len_s * fs)

mfcc_parameters = {
    "fs": fs,
    "n_fft": n_fft,
    "frame_size": n_fft / fs,
    "frame_step": int(n_fft / 3) / fs,
}

def pad_data(x, target_len):
    return librosa.util.fix_length(x, size=target_len, mode="wrap")

def preprocess_audio(path):
    _, data = sci_wav.read(path)

    # pad to fixed length
    data = pad_data(data, fix_len)

    # MFCC extraction
    mfcc = extract_mfcc_feature(
        data=data,
        sample_rate=fs,
        n_fft=n_fft,
        frame_size=mfcc_parameters["frame_size"],
        frame_step=mfcc_parameters["frame_step"],
    )

    # flatten
    X = mfcc.reshape(1, -1)

    # PCA
    return pca.transform(X)

REAL_FOLDER = "data/real"

if __name__ == "__main__":
    wav_files = [f for f in os.listdir(REAL_FOLDER) if f.endswith(".wav")]

    if len(wav_files) == 0:
        print("No wav files found in data/real folder.")
        exit()

    print(f"Found {len(wav_files)} real audio files.\n")

    for wav_file in wav_files:
        full_path = os.path.join(REAL_FOLDER, wav_file)

        X = preprocess_audio(full_path)
        pred = svm.predict(X)[0]
        prob = svm.predict_proba(X)[0]

        label = "bad" if pred == 1 else "good"

        print(f"File: {wav_file}")
        print(f" → Prediction: {label}")
        print(f" → Probabilities: {prob}\n")
