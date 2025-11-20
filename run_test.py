from train import train
import pickle

if __name__ == "__main__":

    svm_model, pca, X_train_pca, y_train, X_test_pca, y_test = train()

    svm_loaded = pickle.load(open("svm_model.pkl", "rb"))
    pca_loaded = pickle.load(open("pca.pkl", "rb"))

    predictions = svm_loaded.predict(X_test_pca[:5])
    probs = svm_loaded.predict_proba(X_test_pca[:5])
    print(f"Predictions from loaded model: {predictions}")
    print(probs)

