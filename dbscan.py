from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt

class MyDBSCAN:    
    def __init__(self, dataset: np.ndarray):
        self.dataset = dataset
        self.eps = None
        self.rho_c = None
        self.prec = None
        self.rhos = None
        self.rhos_cache = None
        self.neighbors = None
        self.labels = None
        self.c_roles = None
        
        
    def densities(self, eps: float):
        #basically we are computing densities with the heaviside kernel
        N=len(self.dataset)
        tree=KDTree(self.dataset)
        self.rhos_cache = np.zeros(N)
        self.eps = eps
        #all neighbors within eps (O(N log N) query)
        self.neighbors = tree.query_ball_point(self.dataset, r=self.eps)
        for i in range(N):
            self.rhos_cache[i] = len(self.neighbors[i])
        
        #normalize densities between 0 and 1
        self.rhos_cache /= N
        
        return None
    
    def adjust_density_precision(self, prec: float = 2**-64):
        #set precision
        self.prec = prec
        
        #adjust the densities to the desired precision s.t. density is between 0 and 1
        lbs = np.clip(self.rhos_cache - self.prec/2, 0, 1)
        ubs = np.clip(self.rhos_cache + self.prec/2, 0, 1)
        
        self.rhos = np.random.uniform(lbs, ubs)
        
        #2nd renormalization
        rescale_factor = np.max(self.rhos_cache)/np.max(self.rhos)
        
        self.rhos *= rescale_factor
        
        return self.rhos
    
  
    def roles(self, rho_c: float):
        self.rho_c = rho_c
        #identify core points    
        self.c_roles = np.where(self.rhos >= self.rho_c, 'core', 'halo')    
            
        return self.c_roles
    
    
    def fit(self):
        N = len(self.dataset)
        visited = np.zeros(N, dtype=bool)
        
        #array to store the labels of each point
        self.labels = np.ones(N, dtype=int) * -1 #-1 means unassigned
        
        core_points = np.where(self.c_roles == 'core')[0]
        
        cid = -1
        for i in core_points:
            if not visited[i]:
                visited[i] = True
                cid += 1
                queue = [i]
                self.labels[i] = cid
                
                while queue:
                    #remove the last element of the queue list with value j
                    j=queue.pop()
                    for n in self.neighbors[j]:
                        if self.labels[n] == -1:
                            self.labels[n] = cid
                            if (self.c_roles[n] == 'core') and not visited[n]:
                                queue.append(n)
                                visited[n] = True
                        
        return self.labels
                    

    def plot_clusters(self, plot_roles=False):
        plt.figure(figsize=(8, 8))
        #define the color map
        num_clusters = np.max(self.labels) + 1
        cmap = plt.colormaps["tab20"].resampled(num_clusters)
        if plot_roles:
            for cid in range(num_clusters):
                cluster_core_points = np.where((self.labels == cid) & (self.c_roles == 'core'))[0]
                cluster_halo_points = np.where((self.labels == cid) & (self.c_roles == 'halo'))[0]
                plt.scatter(self.dataset[cluster_core_points, 0], self.dataset[cluster_core_points, 1], color=cmap(cid), label=f'Cluster {cid+1} core',s=20)
                plt.scatter(self.dataset[cluster_halo_points, 0], self.dataset[cluster_halo_points, 1], color=cmap(cid), label=f'Cluster {cid+1} halo',alpha=0.5,s=10)
                    
            outliers = np.where(self.labels == -1)[0]
            plt.scatter(self.dataset[outliers, 0], self.dataset[outliers, 1], c='red', marker='x', label='Outliers')
            #plt.title('DBSCAN clustering (prec=$2^{{{}}}$)'.format(np.log2(self.prec)))
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.show()
            
        else:
            for cid in range(num_clusters):
                cluster_points = np.where(self.labels == cid)[0]
                plt.scatter(self.dataset[cluster_points, 0], self.dataset[cluster_points, 1], color=cmap(cid), label=f'Cluster {cid+1}',s=20)
                    
            outliers = np.where(self.labels == -1)[0]
            plt.scatter(self.dataset[outliers, 0], self.dataset[outliers, 1], c='red', marker='x', label='Outliers')
            #plt.title('DBSCAN clustering (prec=$2^{{{}}}$)'.format(np.log2(self.prec)))
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.show()