import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2Vec:

    def __init__(self, model_name="facebook/wav2vec2-base-960h", n_clusters=50):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    def extract_wav2vec2_features(self, audio_data):
        inputs = self.processor(
            audio_data, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state

    def generate_pseudo_labels(self, features):
        labels = self.kmeans.fit_predict(features)

        return labels

    def compute_cluster_mean_and_variance(self, features, labels):
        cluster_means = []
        cluster_variances = []

        for cluster_id in range(self.n_clusters):
            cluster_features = features[labels == cluster_id]

            cluster_mean = np.mean(cluster_features, axis=0)
            cluster_variance = np.var(cluster_features, axis=0)

            cluster_means.append(cluster_mean)
            cluster_variances.append(cluster_variance)

        return np.array(cluster_means), np.array(cluster_variances)

    def get_mean_and_variance(self, audio_data):
        features = (
            self.extract_wav2vec2_features(audio_data.squeeze()).squeeze(0).numpy()
        )
        labels = self.generate_pseudo_labels(features)

        return self.compute_cluster_mean_and_variance(features, labels)
