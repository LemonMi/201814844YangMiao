import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
def read_data(datapath):    
    with open(datapath, 'r') as f:
        text_list = []
        text_label = []
        for line in f.readlines():
            dic = json.loads(line.strip())
            text_list.append(dic['text'])
            text_label.append(dic['cluster'])
    return text_list, text_label
def caculate_tfidf_matrix(text_list):
    vectorizer = CountVectorizer(stop_words='english')
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text_list))
    return tfidf.toarray()

def kmeans(tfidf_array, real_label):
    clustering = KMeans(n_clusters=110, random_state=0).fit(tfidf_array)
    predict_label = clustering.labels_
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def affinity_propagation(tfidf_array, real_label):
    clustering = AffinityPropagation(damping=0.95).fit(tfidf_array)
    predict_label = clustering.labels_ 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def mean_shift(tfidf_array, real_label):
    # bandwidth = estimate_bandwidth(tfidf_array, quantile=0.2, n_samples=200)
    clustering = MeanShift(bandwidth=0.5, bin_seeding=True).fit(tfidf_array)
    predict_label = clustering.labels_ 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def spectral_clustering(tfidf_array, real_label):
    clustering = SpectralClustering(n_clusters=110, assign_labels='discretize', random_state=0).fit(tfidf_array)
    predict_label = clustering.labels_ 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def agglomerative_clustering(tfidf_array, real_label, linkage):
    clustering = AgglomerativeClustering(n_clusters=110, linkage=linkage).fit(tfidf_array)
    predict_label = clustering.labels_ 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def dbscan(tfidf_array, real_label):
    clustering = DBSCAN(eps=1.12, min_samples=2).fit(tfidf_array)
    predict_label = clustering.labels_ 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def gaussian_mixture(tfidf_array, real_label):
    clustering = GaussianMixture(n_components=110, covariance_type='diag', random_state=0).fit(tfidf_array)
    predict_label = clustering.predict(tfidf_array) 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score

def birch(tfidf_array, real_label):
    clustering = Birch(n_clusters=110, compute_labels=True).fit(tfidf_array)
    predict_label = clustering.predict(tfidf_array) 
    score = normalized_mutual_info_score(predict_label, real_label)
    return predict_label, score


datapath = './Tweets.txt'
text_list, text_label = read_data(datapath)
matrix = caculate_tfidf_matrix(text_list)

print("K-Means score:%f"%(kmeans(matrix, text_label)[1]))
print("Affinity Propogation score:%f"%(affinity_propagation(matrix, text_label)[1]))
print("Mean-Shift score:%f"%(mean_shift(matrix, text_label)[1]))
print("Spectral Clustering score:%f"%(spectral_clustering(matrix, text_label)[1]))
print("Agglomerative Clustering-ward score:%f"%(agglomerative_clustering(matrix, text_label, 'ward')[1]))
print("Agglomerative Clustering-complete score:%f"%(agglomerative_clustering(matrix, text_label, 'complete')[1]))
print("Agglomerative Clustering-average score:%f"%(agglomerative_clustering(matrix, text_label, 'average')[1]))
print("DBSCAN score:%f"%(dbscan(matrix, text_label)[1]))
print("Gaussian Mixtures score:%f"%(gaussian_mixture(matrix, text_label)[1]))
print("Birch score:%f"%(birch(matrix, text_label)[1]))