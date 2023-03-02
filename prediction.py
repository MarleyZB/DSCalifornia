import joblib
import pickle
import pipeline

def predict(data):
    clf = joblib.load("m_model.pkl")
    full_pipeline = pipeline.fetch_pipeline()
    data_prepared = full_pipeline.transform(data)
    return clf.predict(data_prepared)