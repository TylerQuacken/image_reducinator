import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

class Clusterinator:
    """K-Means clusterer"""

    def __init__(self, n_clusters:int, data:np.ndarray):
        """Initializes a Clusterinator to cluster data into groups
        
        ARGUMENTS
        ---------
        n_clusters [int]: Number of groups in which to cluster data
        data [np.ndarray - shape (batch_dims..., single_array)]: the data to sort. The last dimension is treated as the vector of data, all preprended dimensions are treated as batch dimensions

        RETURNS
        -------
        none
        """
        self.n_clusters = n_clusters
        self.data = data
        self.batch_dims = self.data.shape[:-1]
        self.data_dim = self.data.shape[-1]

        # I want this to do max/min over all axes except the last one
        self.batch_dim_nums = tuple(range(len(self.batch_dims)))
        self.data_max = np.amax(data, axis=self.batch_dim_nums)
        self.data_min = np.amin(data, axis=self.batch_dim_nums)
        self.data_norm = (self.data - self.data_min)/(self.data_max - self.data_min)
        self.cluster_assigns = np.zeros(self.batch_dims, dtype=int)
        # Should change to randomly choose two data points instead of choose two random spots
        self.centroids = np.random.rand(self.n_clusters, self.data_dim)

        # self.cluster_deviations = np.ones(n_clusters)
        self.cluster_deviations = np.array([1,2,1.5])


        self.error = 9999
        self.prev_error = 999999
        self.assign_points_to_clusters()

    def perform_k_means(self, max_iter=100, converge_tol=0.0001):
        converged = False
        i = 0

        while not(converged) and i < max_iter:
            self.plot_data_2d()

            self.calculate_centroids()
            self.assign_points_to_clusters()

            if abs(self.prev_error - self.error) < converge_tol:
                converged = True
                # pass
            
            i += 1


        return converged

    def calculate_centroids(self):
        """Calculates new centroids for each cluster.
        
        ARGUMENTS
        ---------
        none

        RETURNS
        -------
        none
        """

        for i in range(self.n_clusters):
            cluster_i_points = self.cluster_assigns == i
            cluster_i_points = np.expand_dims(cluster_i_points, axis=-1)
            cluster_i_points = np.repeat(cluster_i_points, 2, axis=-1)
            masked_data = np.ma.masked_array(self.data_norm, mask=np.logical_not(cluster_i_points))

            self.centroids[i] = np.mean(masked_data, axis=self.batch_dim_nums)
        

    def assign_points_to_clusters(self):
        self.prev_error = self.error

        norm_expand = np.expand_dims(self.data_norm, -2)
        dist_vecs = self.centroids - norm_expand
        dists = np.linalg.norm(dist_vecs, axis=-1)
        self.error = np.sum(dists**2)
        print(self.error)
        mahalanobis_dists = dists / self.cluster_deviations
        self.cluster_assigns = np.argmin(mahalanobis_dists, axis=-1)


    def plot_data_2d(self):
        plt.clf()
        ax = plt.gca()
        for i in range(self.n_clusters):
            i_points = self.data[self.cluster_assigns == i]
            ax.scatter(i_points[...,0].flatten(), i_points[...,1].flatten())

        
        centroids = self.centroids*(self.data_max - self.data_min) + self.data_min
        ax.scatter(centroids[...,0], centroids[...,1])
        plt.pause(0.03)


if __name__ == "__main__":
    import numpy as np
    from clusterinator.Clusterinator import Clusterinator
    import matplotlib.pyplot as plt

    width = 30
    height = 40
    n_channels = 2
    # data = np.random.randint(0, 255, size=(width, height, n_channels))

    data = np.zeros((width, height, n_channels))
    data[:, :height//2, :] = (np.random.randn(width, height//2, n_channels) + (10*np.random.rand(n_channels) - 1)) * 128
    data[:, height//2:, :] = (np.random.randn(width, height//2, n_channels) + (10*np.random.rand(n_channels) - 1)) * 128
    # data = np.ones((width, height, n_channels))
    n_clusters = 3

    clusterinator = Clusterinator(n_clusters, data)
    clusterinator.perform_k_means()
    clusterinator.plot_data_2d()
    plt.show()