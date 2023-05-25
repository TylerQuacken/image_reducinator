import numpy as np
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
        batch_dim_nums = tuple(range(len(data.shape)))
        self.data_max = np.amax(data, axis=batch_dim_nums)
        self.data_min = np.amin(data, axis=batch_dim_nums)
        self.data_norm = (self.data - self.data_min)/(self.data_max - self.data_min)
        self.cluster_assigns = np.zeros(self.batch_dims + (self.n_clusters,), dtype=int)
        self.centroids = np.random.rand(self.n_clusters, self.data_dim)

        self.assign_points_to_clusters()

    def calculate_centroids(self):
        """Calculates new centroids for each cluster.
        
        ARGUMENTS
        ---------
        none

        RETURNS
        -------
        none
        """
        pass

    def assign_points_to_clusters(self):
        norm_expand = np.expand_dims(self.data_norm, -2)
        dist_vecs = self.centroids - norm_expand
        dists = np.linalg.norm(dist_vecs, axis=-1)
        self.cluster_assigns = np.argmin(dists, axis=-1)

    def plot_data_2d(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(self.n_clusters):
            i_points = self.data[self.cluster_assigns == i]
            ax.scatter(i_points[...,0].flatten(), i_points[...,1].flatten())

        
        centroids = self.centroids*(self.data_max - self.data_min) + self.data_min
        plt.scatter(centroids[...,0], centroids[...,1])
        plt.show()


if __name__ == "__main__":
    import numpy as np
    from clusterinator.Clusterinator import Clusterinator
    import matplotlib.pyplot as plt

    width = 30
    height = 40
    n_channels = 2
    data = np.random.randint(0, 255, size=(width, height, n_channels))
    # data = np.ones((width, height, n_channels))
    n_clusters = 2

    clusterinator = Clusterinator(n_clusters, data)
    clusterinator.plot_data_2d()