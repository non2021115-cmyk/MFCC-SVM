import pickle
import numpy as np

# 1) Load saved SVM model
svm = pickle.load(open("svm_model.pkl", "rb"))

# 2) Load precomputed real data PCA features
real_pca = np.load("real_test_pca.npy")

# 3) Predict
predictions = svm.predict(real_pca)

# Print results
for i, pred in enumerate(predictions):
    label = "bad" if pred == 1 else "good"
    print(f"sample_{i}: {label}, prob={probs[i]}")

