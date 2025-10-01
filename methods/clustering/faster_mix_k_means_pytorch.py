import numpy as np
import copy
import random
from project_utils.cluster_utils import cluster_acc
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
import project_utils.lorentz as L
import project_utils.poincare as P
import torch
import torch.nn.functional as F

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    Distance is not squared, meaning this is actually distance squared
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis

# TODO: Consider doing all this on GPU?
class K_Means:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None,
                 hyperbolic=False, curv = 1.0, poincare=False, cluster_size=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode
        self.hyperbolic = hyperbolic
        self.curv = curv
        self.poincare = poincare
        self.cluster_size = cluster_size

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets

    # Chooses the new centers as the points farthest away from any other centers
    def kpp(self, X, pre_centers=None, k=10, random_state=None, n_iterations = 1000):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers[:]

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])
        current_iterations = 0

        while C.shape[0] < k:

            if self.hyperbolic:
                # TODO: Consider adding batching to Lorentz and Poincare pairwise distance if needed
                if self.poincare:
                    dist = P.pairwise_dist(X, C, curv = -self.curv, eps=1e-6)**2
                else:
                    dist = L.pairwise_dist(X, C, curv = self.curv, eps=1e-6)**2
            else:
                dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
                print(f"K++: No center assigned. Iterating again. Current centers: {C.shape[0]}")
                if current_iterations > n_iterations:
                    print(f"K++: No center assigned after {n_iterations} iterations. Exiting.")
                    return None, False
                current_iterations += 1
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
                C = torch.cat((C, X[ind].view(1, -1)), dim=0)
                current_iterations = 0
                

        return C, True


    def fit_once(self, X, random_state):

        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(len(X))
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == 'k-means++':
            temp_centers, success = self.kpp(X, k=self.k, random_state=random_state)
            if success is False:
                random_state = check_random_state(self.random_state)
                idx = random_state.choice(len(X), self.k, replace=False)
                for i in range(self.k):
                    centers[i] = X[idx[i]]
            else:
                centers = temp_centers

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        #begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):

            centers_old = centers.clone()
            if self.hyperbolic:
                if self.poincare:
                    dist = P.pairwise_dist(X, centers, curv = -self.curv, eps=1e-6)
                else:
                    dist = L.pairwise_dist(X, centers, curv = self.curv, eps=1e-6)
            else:
                dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                if len(selected) == 0:
                    #print("Class number:", idx, "has no samples. Skipping.")
                    continue
                if self.hyperbolic:
                    if self.poincare:
                        centers[idx] = P.einstein_midpoint(selected, -self.curv)
                    else:
                        L_Centroid = L.lorentz_centroid(selected, self.curv)
                        if L_Centroid is None:
                            centers[idx] = L.einstein_midpoint(selected, self.curv)
                        else:
                            centers[idx] = L_Centroid
                else:
                    #centers[idx] = L.einstein_midpoint(selected, self.curv) if self.hyperbolic else selected.mean(dim=0)
                    centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            if self.hyperbolic:
                if self.poincare:
                    center_shift = torch.sum(P.elementwise_dist(centers, centers_old, curv = -self.curv, eps=1e-6))
                else:
                    center_shift = torch.sum(L.elementwise_dist(centers, centers_old, curv = self.curv, eps=1e-6))
            else:
                center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        # DONE: Check if taking the mean is the same in hyperbolic space
        # Checked. It does not, so I am now using Einstein midpoint
        if self.hyperbolic:
        #if False:
            if self.poincare:
                l_centers = torch.stack([P.einstein_midpoint(l_feats[idx_list], -self.curv) for idx_list in support_idxs])
            else:
                temp_centers = []
                for idx_list in support_idxs:
                    L_Centroid = L.lorentz_centroid(l_feats[idx_list], self.curv)
                    if L_Centroid is None:
                        temp_centers.append(L.einstein_midpoint(l_feats[idx_list], self.curv))
                    else:
                        temp_centers.append(L_Centroid)
                l_centers = torch.stack(temp_centers)
                #l_centers = torch.stack([L.lorentz_centroid(l_feats[idx_list], self.curv) for idx_list in support_idxs])
                #l_centers = torch.stack([L.einstein_midpoint(l_feats[idx_list], self.curv) for idx_list in support_idxs])
        else:
            l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        prob_centers = torch.ones([self.k]).type_as(cat_feats).cpu()
        prob_centers_labelled = torch.zeros([self.k]).type_as(cat_feats).cpu()

        centers[:len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers
        if self.init == 'k-means++':
            temp_centers, success = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)
            if success is False:
                print("K++ failed. Using random initialization.")
                random_state = check_random_state(self.random_state)
                idx = random_state.choice(len(u_feats), self.k, replace=False)
                for i in range(self.k):
                    centers[len(l_classes):i] = u_feats[idx[i]]
            else:
                centers = temp_centers

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(u_feats), self.k, replace=False)
            for i in range(self.k):
                centers[len(l_classes):i] = u_feats[idx[i]]

        else:
            for i in range(self.k):
                centers[len(l_classes):i] = u_feats[i]

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        cluster_size=self.cluster_size
        for idx in range(self.k):
            prob_centers[idx] = (labels == idx).float().sum()

        cluster_free=torch.zeros_like(labels[l_num:]).bool()
        sparse_cluster=torch.zeros_like(prob_centers).bool()

        all_center_shifts = []
        for it in range(self.max_iterations):
            for idx in range(self.k):
                prob_centers[idx] = (labels == idx).sum()
            for idx in range(self.k):
                prob_centers_labelled[idx] = (labels[:l_num] == idx).sum()

            centers_old = centers.clone()

            if self.hyperbolic:
                if self.poincare:
                    dist = P.pairwise_dist(u_feats, centers, curv = -self.curv, eps=1e-6)
                else:
                    dist = L.pairwise_dist(u_feats, centers, curv = self.curv, eps=1e-6)
            else:
                dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            if self.hyperbolic:
                # DONE: Optimize by not having to take pairwise distance
                if self.poincare:
                    l_mindist = P.elementwise_dist(l_feats, centers[labels[:l_num]], curv = -self.curv, eps=1e-6)
                else:
                    #l_dist = L.pairwise_dist(l_feats, centers, curv = self.curv)
                    # Get the distance to the corresponding label's center
                    #l_mindist = l_dist[torch.arange(l_num), labels[:l_num]]
                    #l_mindist, _ = torch.min(l_dist, dim=1)
                    l_mindist = L.elementwise_dist(l_feats, centers[labels[:l_num]], curv = self.curv, eps=1e-6)
            else:
                # Done: Investigate how the hell this works. l_feats and centers are not the same shape afaik
                # Might have to do with the assignment inside centers. Just looked, it does
                l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels
            if cluster_size is not None:
                cluster_free_unseen=cluster_free
                if it < self.max_iterations:
                    cluster_free_unseen[0:] = False
                    sparse_cluster[0:] = True
                    # Woah. Double for loop
                    for i in range(it):
                        for idx in range(self.k):
                            u_selected = torch.nonzero(u_labels == idx).squeeze()
                            num = cluster_size - prob_centers_labelled[idx].int()
                            if len(u_selected.size())==0 or u_selected.shape[0]<num:
                                sparse_cluster[idx]=True
                            elif u_selected.shape[0]>num:
                                indexes = torch.argsort(dist[u_selected,idx], dim=0)[num:]
                                indexes =u_selected[indexes]
                                cluster_free_unseen[indexes]=True
                                sparse_cluster[idx]=False
                            else:
                                sparse_cluster[idx]=False
                        cluster_distance=torch.argsort(dist[cluster_free_unseen, :][:, sparse_cluster], dim=1)
                        mid_labels = torch.from_numpy(np.arange(self.k))[sparse_cluster][cluster_distance]
                        if mid_labels.shape[1]!=0:
                            u_labels[cluster_free_unseen] = mid_labels[:, 0]
                            cluster_free_unseen[0:] = False
                labels[l_num:]= u_labels

            for idx in range(self.k):

                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                if len(selected) == 0:
                    #print("Class number:", idx, "has no samples. Skipping.")
                    continue
                if self.hyperbolic:
                    if self.poincare:
                        centers[idx] = P.einstein_midpoint(selected, -self.curv)
                    else:
                        L_Centroid = L.lorentz_centroid(selected, self.curv)
                        if L_Centroid is None:
                            centers[idx] = L.einstein_midpoint(selected, self.curv)
                        else:
                            centers[idx] = L_Centroid
                else:
                    #centers[idx] = L.einstein_midpoint(selected, self.curv) if self.hyperbolic else selected.mean(dim=0)
                    centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            # DONE: Add hyperbolic distance
            if self.hyperbolic:
                if self.poincare:
                    center_shift = torch.sum(P.elementwise_dist(centers, centers_old, curv = -self.curv, eps=1e-6))
                else:
                    center_shift = torch.sum(L.elementwise_dist(centers, centers_old, curv = self.curv, eps=1e-6))
            else:
                center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            all_center_shifts.append(center_shift.item())
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        # For some reason the SelEx code just runs the loop again but without balancing
        for it in range(self.max_iterations):
            for idx in range(self.k):
                prob_centers[idx] = (labels == idx).sum()
            for idx in range(self.k):
                prob_centers_labelled[idx] = (labels[:l_num] == idx).sum()

            centers_old = centers.clone()

            if self.hyperbolic:
                if self.poincare:
                    dist = P.pairwise_dist(u_feats, centers, curv = -self.curv, eps=1e-6)
                else:
                    dist = L.pairwise_dist(u_feats, centers, curv = self.curv, eps=1e-6)
            else:
                dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            if self.hyperbolic:
                # DONE: Optimize by not having to take pairwise distance
                if self.poincare:
                    l_mindist = P.elementwise_dist(l_feats, centers[labels[:l_num]], curv = -self.curv, eps=1e-6)
                else:
                    #l_dist = L.pairwise_dist(l_feats, centers, curv = self.curv)
                    # Get the distance to the corresponding label's center
                    #l_mindist = l_dist[torch.arange(l_num), labels[:l_num]]
                    #l_mindist, _ = torch.min(l_dist, dim=1)
                    l_mindist = L.elementwise_dist(l_feats, centers[labels[:l_num]], curv = self.curv, eps=1e-6)
            else:
                # Done: Investigate how the hell this works. l_feats and centers are not the same shape afaik
                # Might have to do with the assignment inside centers. Just looked, it does
                l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):

                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                if len(selected) == 0:
                    #print("Class number:", idx, "has no samples. Skipping.")
                    continue
                if self.hyperbolic:
                    if self.poincare:
                        centers[idx] = P.einstein_midpoint(selected, -self.curv)
                    else:
                        L_Centroid = L.lorentz_centroid(selected, self.curv)
                        if L_Centroid is None:
                            centers[idx] = L.einstein_midpoint(selected, self.curv)
                        else:
                            centers[idx] = L_Centroid
                else:
                    #centers[idx] = L.einstein_midpoint(selected, self.curv) if self.hyperbolic else selected.mean(dim=0)
                    centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            # DONE: Add hyperbolic distance
            if self.hyperbolic:
                if self.poincare:
                    center_shift = torch.sum(P.elementwise_dist(centers, centers_old, curv = -self.curv, eps=1e-6))
                else:
                    center_shift = torch.sum(L.elementwise_dist(centers, centers_old, curv = self.curv, eps=1e-6))
            else:
                center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            all_center_shifts.append(center_shift.item())
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1, all_center_shifts


    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


    def fit_mix(self, u_feats, l_feats, l_targets):
        if isinstance(u_feats, np.ndarray):
            u_feats = torch.from_numpy(u_feats)
        if isinstance(l_feats, np.ndarray):
            l_feats = torch.from_numpy(l_feats)
        if isinstance(l_targets, np.ndarray):
            l_targets = torch.from_numpy(l_targets)
        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):

                labels, inertia, centers, n_iters, center_shifts = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
                    self.center_shifts_ = center_shifts

        else:

            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
                                                              for seed in seeds)
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters, center_shifts = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]
            self.center_shifts_ = center_shifts[best]


def main():

    import matplotlib.pyplot as plt
    from matplotlib import style
    import pandas as pd
    style.use('ggplot')
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility
    print(torch.from_numpy(X).norm(dim=-1).max())
    X = torch.from_numpy(X)
    #X = torch.where(X.norm(dim=-1, keepdim=True) < 2, X, 2*F.normalize(X, dim=-1))
    X = L.exp_map0(X, curv=torch.tensor(2.0))
    #X = P.expmap0(X, k=torch.tensor(-2.0), project=False)
    #X = P.project(X, k=torch.tensor(-2.0), eps=1-1e-3)
    if X.isnan().any():
        print("X has NaN values")
        exit()
    X = np.array(X)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #  X = torch.from_numpy(X).float().to(device)


    y = np.array(y)
    l_targets = y[y>1]
    l_feats = X[y>1]
    u_feats = X[y<2]
    cat_feats = np.concatenate((l_feats, u_feats))
    y = np.concatenate((y[y>1], y[y<2]))
    cat_feats = torch.from_numpy(cat_feats).to(device)
    u_feats = torch.from_numpy(u_feats).to(device)
    l_feats = torch.from_numpy(l_feats).to(device)
    l_targets = torch.from_numpy(l_targets).to(device)

    #km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10)
    km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10, hyperbolic=True, curv=2.0)
    #km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10, hyperbolic=True, poincare=True, curv=2.0)

    #  km.fit(X)

    km.fit_mix(u_feats, l_feats, l_targets)
    #  X = X.cpu()
    X = cat_feats.cpu()
    centers = km.cluster_centers_.cpu()
    print(centers)
    pred = km.labels_.cpu()
    print('nmi', nmi_score(pred, y))

    # Plotting starts here
    colors = 10*["g", "c", "b", "k", "r", "m"]

    for i in range(len(X)):
        x = X[i]
        plt.scatter(x[0], x[1], color = colors[pred[i]],s = 10)

    for i in range(4):
        plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')

    plt.title(f"nmi={nmi_score(pred, y)}")
    #plt.savefig('kmeans.png', dpi=300)
    plt.savefig('hyperbolic_kmeans.png', dpi=300)
    #plt.savefig('poincare_kmeans.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()