import unittest
import torch
from ..prior import Wav2Vec2
import numpy as np

class TestWav2Vec2(unittest.TestCase):

    def setUp(self):
        self.wav2vec2 = Wav2Vec2(model_name='facebook/wav2vec2-base-960h', n_clusters=5)

        self.audio_data = torch.randn(16000) 
        self.features = torch.randn(100, 768)  
        self.labels = self.wav2vec2.generate_pseudo_labels(self.features)

    def test_extract_wav2vec2_features(self):
        features = self.wav2vec2.extract_wav2vec2_features(self.audio_data) 
        self.assertEqual(features.shape, (1, 49, 768))  

    def test_generate_pseudo_labels(self):
        labels = self.wav2vec2.generate_pseudo_labels(self.features)
        self.assertEqual(len(labels), 100)  
        self.assertTrue(np.all(np.isin(labels, range(self.wav2vec2.n_clusters))))  


    def test_compute_cluster_mean_and_variance(self):
        cluster_means, cluster_variances = self.wav2vec2.compute_cluster_mean_and_variance(self.features.numpy(), self.labels)
        self.assertEqual(cluster_means.shape, (self.wav2vec2.n_clusters, self.features.shape[1])) 
        self.assertEqual(cluster_variances.shape, (self.wav2vec2.n_clusters, self.features.shape[1]))

    def test_random_audio_clustering(self):
        num_audio_samples = 200
        audio_length = 16000
        random_audio_data = [torch.randn(audio_length).numpy() for _ in range(num_audio_samples)]

        all_features = []
        for audio in random_audio_data:
            features = self.wav2vec2.extract_wav2vec2_features(audio)
            all_features.append(features.squeeze(0).mean(dim=0).numpy())  

        all_features = np.vstack(all_features)  

        labels = self.wav2vec2.generate_pseudo_labels(all_features)

        cluster_means, cluster_variances = self.wav2vec2.compute_cluster_mean_and_variance(all_features, labels)

        self.assertEqual(cluster_means.shape, (self.wav2vec2.n_clusters, all_features.shape[1]))
        self.assertEqual(cluster_variances.shape, (self.wav2vec2.n_clusters, all_features.shape[1])) 

        for i in range(self.wav2vec2.n_clusters):
            print(f"Cluster {i}:")
            print(f"Mean shape: {cluster_means[i].shape}")
            print(f"Variance shape: {cluster_variances[i].shape}") 

if __name__ == "__main__":
    unittest.main()
