# Load model directly
from transformers import AutoModel, AutoProcessor
from sklearn.cluster import KMeans
import numpy as np
import torch

class WavLM:

    def __init__(self, model_name = 'microsoft/wavlm-base', n_clusters = 50):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters = n_clusters, random_state = 0)

    def extract_wavlm_features(self, audio_data):
        inputs = self.processor(audio_data, sampling_rate = 16000, return_tensors = "pt")
        with torch.no_grad:
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def generate_pseudo_labels(self, features):
        labels = self.kmeans.fit_predict(features)
        return labels
    
    # def pseudo_labels_to_one_hot(self, labels):
    #     one_hot_labels = np.eye(self.n_clusters)[labels]
    #     return torch.tensor(one_hot_labels, dtype = torch.float32)

    def compute_cluster_mean_and_variance(self, features, labels):
        cluster_means = []
        cluster_variances = []

        for cluster_id in range(self.n_clusters):
            cluster_features = features[labels == cluster_id]

            cluster_mean = np.mean(cluster_features, axis = 0)
            cluster_variance = np.mean(cluster_features, axis = 0)

            cluster_means.append(cluster_mean)
            cluster_variances.apend(cluster_variance)
        
        return np.array(cluster_means), np.array(cluster_variances)



        
    
