from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt

class DPClustering:
    """
    Density-Peak Clustering (DPC) algorithm implementation.
    
    To initialize the class, provide a 2D dataset as a numpy array of shape (N, 2), being N the number of data points.
    
    Clusters data based on DPC (Gaussian- or Heaviside-based), given specific thresholds.
    The class provides the following methods:
    
        - computes densities (rhos) given the scale parameter (dc) using:
            - Gaussian kernel -> gaussian_density(dc)
                - inputs: dc : float
                - outputs: None
            - Heaviside kernel -> heaviside_density(dc)
                - inputs: dc : float
                - outputs: None
                
        - special method to generate random densities -> random_density()
            - inputs: None
            - outputs: rhos : np.ndarray
            Details: For some case study it could be important to evaluate how the DPC works with random densities. This method generates random densities uniformly
            distributed between 0 and 1. Returns a np.ndarray of shape (N,), where the i-th entry corresponds to a float value with the density of the i-th data point.
            After this method, it is not supposed to call 'adjust_density_precision'.
            
                
        - adjusts the densities to the desired precision (prec) -> adjust_density_precision(prec)
            - inputs: prec : float (default: 2**-64)
            - outputs: rhos : np.ndarray
            Details: Add a random noise within the interval [-prec/2, +prec/2] to the density value computed by the density method. Since it is required the density to be
            between 0 and 1, the adjustment is truncated to be within [0, 1] Returns a np.ndarray of shape (N,), where the i-th entry corresponds to a float value with 
        the adjusted density of tje i-th data point.
            
        - computes nearest-higher indices (nh) and distances (deltas) -> nearest_highers()
            - inputs: None
            - outputs: nh : np.ndarray, deltas : np.ndarray
            Details: Computes the nearest-higher of each point in the dataset, i.e., the index of the closest point with a higher density than the
        point itself. Returns a np.ndarray of shape (N,), where the i-th entry corresponds to an integer value with the index of the nearest-higher of
        the i-th data point. The entry corresponding the the point with the highest density is -1. Also computes the distance to the
        nearest-higher point, returning a np.ndarray of shape (N,), in which the i-th entry corresponds to a float value with the distance to the nearest-higher of
        the i-th data point. If the point is a root, the value is set to the maximum distance between any two points in the dataset plus one.
            
        - finds the roots (roots) given thresholds (rho_c, delta_c) -> find_roots(rho_c, delta_c)
            - inputs: rho_c : float, delta_c : float
            - outputs: roots : np.ndarray
            Details: Finds the roots of the clusters, i.e., the points with density > rho_c and distance to nearest-higher > delta_c. Returns a np.ndarray of shape (M,),
            where M is the number of roots found, and each entry corresponds to an integer value with the index of a root.
            
        - constructs the clusters (clusters) given thresholds (rho_c, delta_c) -> fit(rho_c, delta_c, mark_outliers)
            - inputs: rho_c : float, delta_c : float, mark_outliers : bool (default: True)
            - outputs: clusters_labels : np.ndarray, outliers : np.ndarray
            Details: Constructs the clusters by assigning each point to the same cluster as its nearest-higher, starting from the roots found with 'find_roots'.
        If mark_outliers is True, points with density <= rho_c and distance to nearest-higher > delta_c are marked as outliers, otherwise the outliers np.ndarray
        is left empty. Returns a np.ndarray of shape (N,), in which each entry corresponds to the dataset point and contains the label cluster to which that point belongs. Also returns a np.ndarray
        of shape (L,), where L is the number of outliers found, and each entry corresponds to an integer value with the index of an outlier.
            
        - computes the role of each point in the dataset (core or halo) -> roles()
            - inputs: None
            - outputs: clusters_roles : np.ndarray
            Details: The border of a cluster is defined as the set of points that are within a distance dc from points belonging to other clusters. The core of a cluster is defined
            as the set of points with density higher than the maximum density of the its border. The rest of the points in a cluster are classified as halo points. Returns a .ndarrays
            of shape (N,), where each entry corresponds to the dataset point and contains a string value with the role of that point: 'core' or 'halo'.
            
        - plots the nearest-higher tree -> plot_tree()
            - inputs: None
            - outputs: None
            Details: Plots the nearest-higher tree, where each point is connected to its nearest-higher point with a line. The roots are highlighted in red.
            
        - plots the decision graphs -> plot_decision_graphs(rho_c, delta_c)
            - inputs: rho_c : float, delta_c : float
            - outputs: None
            Details: Plots the decision graphs: delta vs rho and gamma vs n, where gamma = rho * delta. The thresholds rho_c and delta_c are indicated with dashed lines.
            The roots are annotated with their respective indices.
            
        - plots the clusters -> plot_clusters(core=None, halo=None)
            - inputs: core : list of np.ndarray (default: None), halo : list of np.ndarray (default: None)
            - outputs: None
            Details: Plots the clusters, with different colors for each cluster. If core and halo are provided, core points are plotted with full opacity
            and halo points with reduced opacity. The roots are marked with a black 'x' and outliers (if any) are marked in black. 
        
    Attributes:
        dataset : np.ndarray
        dc : float
        prec : float
        rhos_cached : np.ndarray
        rhos : np.ndarray
        nh : np.ndarray
        deltas : np.ndarray
        clusters_labels : np.ndarray
        clusters_roles : np.ndarray
        roots : np.ndarray
        outliers : np.ndarray
        
    Methods:
        gaussian_density(dc)
        heaviside_density(dc)
        random_density()
        adjust_density_precision(prec)
        nearest_highers()
        find_roots(rho_c, delta_c)
        fit(rho_c, delta_c, mark_outliers)
        roles()
        plot_tree()
        plot_decision_graphs(rho_c, delta_c)
        plot_clusters(core, halo)
    """
    
    def __init__(self, dataset: np.ndarray):
        self.dataset = dataset
        self.dc = None
        self.prec = None
        self.rhos_cached = None
        self.rhos = None
        self.nh = None
        self.deltas = None
        self.clusters_labels = None
        self.clusters_roles = None
        self.roots = np.array([], dtype=int)
        self.outliers = np.array([], dtype=int)
        
    def gaussian_density(self, dc: float):
        #set scale factgor
        self.dc = dc
            
        N = len(self.dataset)
        self.rhos_cached = np.zeros(N)
        
        #compute the density of the each element 
        for i in range(N):
            dists2_i = (self.dataset[:,0]-self.dataset[i,0])**2 + (self.dataset[:,1]-self.dataset[i,1])**2
            self.rhos_cached[i] = np.sum(np.exp(-dists2_i/self.dc**2)) / N #normalized
        
        return None
    
    def heaviside_density(self, dc: float):
        #set scale factor
        self.dc = dc
        
        N = len(self.dataset)
        self.rhos_cached = np.zeros(N)
        
        #compute the density of the each element 
        for i in range(N):
            dists2_i = (self.dataset[:,0]-self.dataset[i,0])**2 + (self.dataset[:,1]-self.dataset[i,1])**2
            self.rhos_cached[i] = np.sum(dists2_i <= self.dc**2) / N #normalized
            
        return None
    
    def random_density(self):
        N = len(self.dataset)
        #create random densities uniformly distributed between 0 and 1
        self.rhos  = np.random.rand(N)
        return self.rhos
    
    def adjust_density_precision(self, prec: float = 2**-64):
        #set precision
        self.prec = prec
        
        #adjust the densities to the desired precision s.t. density is between 0 and 1
        lbs = np.clip(self.rhos_cached - self.prec/2, 0, 1)
        ubs = np.clip(self.rhos_cached + self.prec/2, 0, 1)
        
        self.rhos = np.random.uniform(lbs, ubs)
        
        #2nd renormalization
        rescale_factor = np.max(self.rhos_cached)/np.max(self.rhos)
        self.rhos *= rescale_factor
        
        return self.rhos
    
    def nearest_highers(self):
        self.nh = np.zeros_like(self.rhos, dtype=int)   
        self.deltas = np.zeros_like(self.rhos)
        
        indices = np.argsort(self.rhos)
        for k,i in enumerate(indices[:-1]):
            js = indices[:k+1] #elements with smaller density including the point itself
            dists2_i = (self.dataset[:,0]-self.dataset[i,0])**2 + (self.dataset[:,1]-self.dataset[i,1])**2
            dists2_i[js] = np.inf
            self.nh[i] = np.argmin(dists2_i)
            self.deltas[i] = np.sqrt(dists2_i[self.nh[i]])
            #print(self.nh[i],self.deltas[i])
            
        self.nh[indices[-1]] = -1
        self.deltas[indices[-1]] = np.max(self.deltas) + 1
        
        return self.nh, self.deltas
    
    def find_roots(self,rho_c: float, delta_c: float,):
        self.roots = np.where((self.rhos > rho_c) & (self.deltas > delta_c))[0]
        return self.roots
    
    def fit(self, rho_c: float, delta_c: float, mark_outliers: bool = True):
        #array to store the labels of each point
        self.clusters_labels=np.ones_like(self.rhos, dtype=int) * -1 #-1 means unassigned
        #roots found with function 'find_roots'
        if mark_outliers:
            #find outliers
            self.outliers = np.where((self.rhos <= rho_c) & (self.deltas > delta_c))[0]
            
        N = len(self.nh)
        visited = np.zeros(N, dtype=bool)
        visited[self.roots] = True
        visited[self.outliers] = True
        
        for cid,root in enumerate(self.roots):
            cluster=[root]
            frontier = np.array([root])
            
            while len(frontier) > 0:
                #find points whose nearest-higher is in current frontier
                mask = np.isin(self.nh, frontier)
                #filter visited points
                add = np.where(mask & ~visited)[0]           
                if add.size == 0: 
                    break
                #updated the visisted elements
                visited[add] = True
                #add the points to the respective clusters
                cluster.extend(add.tolist())
                #update the frontier
                frontier = add
            
            #add the cluster to the clusters list
            self.clusters_labels[cluster] = cid
            
        return self.clusters_labels, self.outliers
    
    def roles(self):
        num_clusters = len(self.roots)
        N = len(self.rhos)
        
        #the border of a cluster is define as the set of points that are within a distance dc from points belonging to other clusters
        self.clusters_roles = np.full(N, fill_value='halo', dtype=object) #default role is halo
        
        #build tree once for the whole dataset (O(N log N) build)
        tree = KDTree(self.dataset)
        #all neighbors within dc (O(N log N) query)
        neighbors = tree.query_ball_point(self.dataset, r=self.dc)
        
        #detect border points
        for i in range(N):
            cid = self.clusters_labels[i]
            #points withing a distance equal or less than dc from point i (including itself)
            neigh = neighbors[i]
            if  np.any(self.clusters_labels[neigh] != cid):
                #add point to border
                self.clusters_roles[i] = 'border'
                
        #detect core points
        for cid in range(num_clusters):
            cluster_points = np.where(self.clusters_labels == cid)[0]
            border_points = np.where((self.clusters_labels == cid) & (self.clusters_roles == 'border'))[0]
            rho_b = np.max(self.rhos[border_points]) if len(border_points) > 0 else -np.inf
            core_points = cluster_points[self.rhos[cluster_points] > rho_b]
            self.clusters_roles[core_points] = 'core'
            
        #update border to halo
        border_points = np.where(self.clusters_roles == 'border')[0]
        self.clusters_roles[border_points] = 'halo'
            
        return self.clusters_roles
    
    def plot_decision_graphs(self, rho_c: float, delta_c: float):
        fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(20,7))
        axs = axs.flatten()
        #plot delta vs rho
        ax0 = axs[0]
        ax0.set_title(r'$\delta$ vs. $\rho$')
        ax0.plot(self.rhos,self.deltas,'.')
        ax0.set_xlabel('ρ')
        ax0.set_ylabel('δ')
        ax0.axvline(x=rho_c,linestyle='--',color='red',label='$\\rho_c$={:.3f}'.format(rho_c))
        ax0.axhline(y=delta_c,linestyle='--',color='red',label='$\\delta_c$={:.3f}'.format(delta_c))
        ax0.legend(loc='upper left')

        #plot gamma = rho * delta
        ax1 = axs[1]
        ax0.set_title(r'$\gamma$ vs. $n$')
        gammas = self.rhos * self.deltas
        #ids = np.argsort(gammas)
        ax1.plot(gammas,'--.',markersize=8,linewidth=0.7)
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'$\gamma = \rho \delta$')
        
        for i in self.roots:
            ax0.annotate(i,(self.rhos[i],self.deltas[i]))
            ax1.annotate(i,(i,gammas[i]))
      
    def plot_tree(self):
        # nearest-higher tree
        plt.figure(figsize=(8, 8))
        for i, j in enumerate(self.nh):
            if j != -1:
                xs = [self.dataset[i, 0], self.dataset[j, 0]]
                ys = [self.dataset[i, 1], self.dataset[j, 1]]
                plt.plot(xs, ys, 'k-', lw=1)
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=20, alpha=0.6, color='black')
        plt.scatter(self.dataset[self.roots, 0], self.dataset[self.roots, 1], s=50, c='red',label='roots', zorder=4)
        for root in self.roots:
            plt.annotate("{}".format(root), xy=(self.dataset[root,0],self.dataset[root,1]),
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.title('Nearest-higher tree')
        plt.legend()
        plt.show()

    def plot_clusters(self, plot_roles=False):
        plt.figure(figsize=(8, 8))
        #define the color map
        cmap = plt.colormaps["tab20"].resampled(len(self.roots))
        
        if plot_roles:
            halo_points = np.where(self.clusters_roles == 'halo')[0]
            for cid, root in enumerate(self.roots):
                core_points = np.where((self.clusters_labels == cid) & (self.clusters_roles == 'core'))[0]
                plt.scatter(self.dataset[core_points, 0], self.dataset[core_points, 1], color=cmap(cid), label=f'Cluster {cid+1} Core', s=20)
                # mark root
                plt.scatter(self.dataset[root, 0], self.dataset[root, 1], marker='x', s=100, c='black', label=f'Root {root}')
                
            plt.scatter(self.dataset[halo_points, 0], self.dataset[halo_points, 1], color='black', alpha=0.5, label='Halo points', s=10)
        else:
            for cid, root in enumerate(self.roots):
                cluster_points = np.where(self.clusters_labels == cid)[0]
                plt.scatter(self.dataset[cluster_points, 0], self.dataset[cluster_points, 1], color=cmap(cid), label=f'Cluster {cid+1}',s=20)
                # mark root
                plt.scatter(self.dataset[root, 0], self.dataset[root, 1], marker='x', s=100, c='black', label=f'Root {root}')
                
        plt.scatter(self.dataset[self.outliers, 0], self.dataset[self.outliers, 1], c='red', marker='.', label='Outliers')
        plt.title('DPC clustering (prec=$2^{{{}}}$)'.format(np.log2(self.prec)))
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()