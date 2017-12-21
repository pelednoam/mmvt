# http://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
import os.path as op
# import hdbscan
try:
    import debacl
    import fastcluster
except:
    pass
import sklearn.cluster
import scipy.cluster
import sklearn.datasets
import pandas as pd
import time
import numpy as np
import sklearn.cluster as cluster
from sklearn import mixture
import traceback
import inspect

from src.misc.dell import find_electrodes_in_ct as dell
from src.utils import utils

def benchmark_algorithm(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=2):

    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters)

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                               result.flatten()]).T, columns=['x','y'])
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                   result.flatten()]).T, columns=['x','y'])


def plot_clusters(electrodes, algorithm, algorithm_name, output_fol, args, kwds):
    try:
        if inspect.isclass(algorithm):
            if algorithm_name.startswith('GMM'):
                gmm = algorithm(*args, **kwds)
                gmm.fit(electrodes)
                electrodes_groups = gmm.predict(electrodes)
            else:
                electrodes_groups = algorithm(*args, **kwds).fit_predict(electrodes)
        elif inspect.isfunction(algorithm):
            groups_centroids, _ = algorithm(electrodes, *args, **kwds)
    except:
        err = traceback.format_exc()
        print(err)
        print(f'Error with {algorithm_name}')
        return
    groups_num = len(set(electrodes_groups))
    groups = [[] for _ in range(groups_num)]
    for ind, elc_group in enumerate(electrodes_groups):
        groups[elc_group].append(ind)
    print('{}: {} leads were found, with {} electrodes'.format(algorithm_name, groups_num, [len(g) for g in groups]))
    groups_colors = dell.dist_colors(groups_num)
    electrodes_colors = [groups_colors[electrodes_groups[elc_ind]] for elc_ind in range(len(electrodes))]
    utils.plot_3d_scatter(electrodes, colors=electrodes_colors, fname=op.join(output_fol, '{}.png'.format(algorithm_name)))


def init_algs(clusters_num):
    scipy_k_means_data = benchmark_algorithm(dataset_sizes,
                                             scipy.cluster.vq.kmeans, (clusters_num,), {})

    scipy_single_data = benchmark_algorithm(dataset_sizes,
                                            scipy.cluster.hierarchy.single, (), {})

    fastclust_data = benchmark_algorithm(dataset_sizes,
                                         fastcluster.linkage_vector, (), {})

    hdbscan_ = hdbscan.HDBSCAN()
    hdbscan_data = benchmark_algorithm(dataset_sizes, hdbscan_.fit, (), {})

    debacl_data = benchmark_algorithm(dataset_sizes,
                                      debacl.geom_tree.geomTree, (5, 5), {'verbose': False})




def compare(data, n_groups, output_fol):
    # plot_clusters(data.astype(np.float), scipy.cluster.vq.kmeans, 'scipy.cluster.vq.kmeans', output_fol, (n_groups,), {})
    plot_clusters(data, cluster.KMeans, 'KMeans', output_fol, (), {'n_clusters': n_groups})
    for ct in ['spherical', 'tied', 'diag', 'full']:
        plot_clusters(data, mixture.GaussianMixture, 'GMM_{}'.format(ct), output_fol, (),
                      {'n_components': n_groups, 'covariance_type': ct})
    plot_clusters(data, cluster.AffinityPropagation, 'AffinityPropagation', output_fol, (), {'preference': -5.0, 'damping': 0.95})
    plot_clusters(data, cluster.MeanShift, 'MeanShift', output_fol, (0.175,), {'cluster_all': False})
    plot_clusters(data, cluster.SpectralClustering, 'SpectralClustering', output_fol, (), {'n_clusters': n_groups})
    plot_clusters(data, cluster.AgglomerativeClustering, 'AgglomerativeClustering', output_fol, (), {'n_clusters': n_groups, 'linkage': 'ward'})
    plot_clusters(data, cluster.DBSCAN, 'DBSCAN', output_fol, (), {'eps': 0.025})
    # plot_clusters(data, hdbscan.HDBSCAN, 'HDBSCAN', output_fol, (), {'min_cluster_size': 15})


if __name__ == '__main__':
    root = [f for f in ['/home/npeled/Documents', '/homes/5/npeled/space1/Documents'] if op.isdir(f)][0]
    fol = op.join(root, 'finding_electrodes_in_ct', 'comparisson')
    data_fname = op.join(fol, 'af452', 'objects.pkl')
    (subject, electrodes, groups, org_groups, groups_hemis, ct_electrodes, ct_voxels, threshold, n_components,
     n_groups) = utils.load(data_fname)
    compare(ct_electrodes, n_groups, fol)
    print('Cylinders: {} leads were found, with {} electrodes'.format(len(groups), [len(g) for g in groups]))
    dell.plot_groups(electrodes, org_groups, output_fol=fol, image_name='Cylinders')
    print('finish!')