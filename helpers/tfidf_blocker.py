from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import hnswlib
import numpy as np
import os
import fastcluster
import hcluster


def vectorize(string_list, ngram_range, analyzer, **kwargs):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer, **kwargs)
    tfidf_matrix = vectorizer.fit_transform(string_list)
    return tfidf_matrix


def index_on_approx_knn(tfidf_dense_matrix, ef_construction=100, M=50):
    approx_knn_index = hnswlib.Index(space='cosine', dim=tfidf_dense_matrix.shape[1])
    approx_knn_index.set_num_threads(os.cpu_count())
    # https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md#construction-parameters
    approx_knn_index.init_index(
        max_elements=tfidf_dense_matrix.shape[0],
        ef_construction=ef_construction,
        M=M)
    approx_knn_index.set_ef(ef_construction)
    approx_knn_index.add_items(tfidf_dense_matrix)
    return approx_knn_index


def _hnswlib_result_to_csr_matrix(neighbor_array, similarity_array):
    row = np.repeat(np.arange(neighbor_array.shape[0]), neighbor_array.shape[1])
    col = neighbor_array.flatten()
    data = similarity_array.flatten()
    return csr_matrix((data, (row, col)), shape=(neighbor_array.shape[0], neighbor_array.shape[0]))


def compute_knn_similarity_matrix(approx_knn_index, tfidf_dense_matrix, k, threshold):
    neighbor_array, distance_array = approx_knn_index.knn_query(tfidf_dense_matrix, k=k)
    similarity_array = 1 - distance_array
    del distance_array
    similarity_array[similarity_array < threshold] = 0
    pairwise_similarity_csr_matrix = _hnswlib_result_to_csr_matrix(neighbor_array, similarity_array)
    return pairwise_similarity_csr_matrix


def cluster_csr_distance_matrix(pairwise_distance_dense_matrix, threshold, linkage='complete'):
    labels = AgglomerativeClustering(
        affinity='precomputed',
        linkage=linkage,
        distance_threshold=1 - threshold,
        n_clusters=None
    ).fit_predict(pairwise_distance_dense_matrix)
    return labels


def block_with_tfidf_ann(string_list, threshold, k=10):
    tfidf_matrix = vectorize(string_list, ngram_range=(2, 2), analyzer='char_wb')
    tfidf_dense_matrix = tfidf_matrix.toarray()  # hnswlib needs dense matrix
    del tfidf_matrix
    approx_knn_index = index_on_approx_knn(tfidf_dense_matrix)
    pairwise_similarity_csr_matrix = compute_knn_similarity_matrix(approx_knn_index, tfidf_dense_matrix, k=k, threshold=threshold)
    del tfidf_dense_matrix, approx_knn_index
    pairwise_distance_dense_matrix = 1 - pairwise_similarity_csr_matrix.toarray()
    del pairwise_similarity_csr_matrix
    labels = cluster_csr_distance_matrix(pairwise_distance_dense_matrix, threshold=threshold)
    return labels


def block_with_tfidf_brute(string_list, threshold, k=10, linkage='complete'):
    tfidf_matrix = vectorize(string_list, ngram_range=(2, 2), analyzer='char_wb')
    pairwise_similarity_csr_matrix = awesome_cossim_topn(
        tfidf_matrix,
        tfidf_matrix.T,
        ntop=k,
        lower_bound=threshold,
        use_threads=True,
        n_jobs=os.cpu_count())
    pairwise_distance_dense_matrix = 1 - pairwise_similarity_csr_matrix.toarray()
    labels = AgglomerativeClustering(
        affinity='precomputed',
        linkage=linkage,
        distance_threshold=1 - threshold,
        n_clusters=None
    ).fit_predict(pairwise_distance_dense_matrix)
    return labels
