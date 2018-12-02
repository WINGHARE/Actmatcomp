"""

Author: CHEN junyi junyichen8-c@my.cityu.edu.hk

License: BSD 3 clause
"""

import os
import string
import sys
import random
import logging

import numpy as np
import pandas
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors
from sklearn.utils import resample
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer,LabelEncoder

class Baseclustering:
    def __init__(self):
        self.metric = 'node_1nn'
        self.metric_dic = {
            'node_1nn': self.exp_dis,
            'boost_node_1nn': self.exp_dis_boost,
            'single_link': self.exp_cdist,
            'group_average': self.exp_avgdist,
            'complete_link': self.exp_maxdist,
            'min_cdist': self.exp_axis_min_dist
        }
        self.node_to_exp = dict()
        self.thedata = np.array([])
        self.theindices = np.array([])

    # Following are some distances metrics between nodes

    def exp_dis(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]

        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        # If sample > 500 do sampling
        if len(n1_samples) > num_samps:
            n1_samples = fast_sampling(n1_samples, num=num_samps, seed=0)
        if len(n2_samples) > num_samps:
            n2_samples = fast_sampling(n2_samples, num=num_samps, seed=0)

        lblen = num_samps

        samples = np.append(
            self.thedata[n1_samples], self.thedata[n2_samples], axis=0)
        dism = squareform(pdist(samples, 'euclidean'))
        np.fill_diagonal(dism, float('inf'))

        neighbout_idx = dism.argmin(axis=0)

        c1_vote = sum(neighbout_idx[:lblen] < lblen)
        c2_vote = sum(neighbout_idx[lblen:] >= lblen)

        return (c1_vote + c2_vote) / (len(neighbout_idx))

    def exp_dis_boost(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]

        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        result = 0

        for i in range(0, 5):

            # If sample > 500 do sampling
            if len(n1_samples) > num_samps:
                # , random_state=0)
                n1_samples = resample(
                    n1_samples, replace=False, n_samples=num_samps)
            if len(n2_samples) > num_samps:
                # , random_state=0)
                n2_samples = resample(
                    n2_samples, replace=False, n_samples=num_samps)

            lblen = num_samps

            samples = np.append(
                self.thedata[n1_samples], self.thedata[n2_samples], axis=0)
            dism = squareform(pdist(samples, 'euclidean'))
            np.fill_diagonal(dism, float('inf'))

            neighbout_idx = dism.argmin(axis=0)

            c1_vote = sum(neighbout_idx[:lblen] < lblen)
            c2_vote = sum(neighbout_idx[lblen:] >= lblen)

            result += (c1_vote + c2_vote) / (len(neighbout_idx))

        return result / 5

    def exp_cdist(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]

        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        # If sample > 500 do sampling
        if len(n1_samples) > num_samps:
            n1_samples = resample(n1_samples, replace=False,
                                  n_samples=num_samps, random_state=0)
        if len(n2_samples) > num_samps:
            n2_samples = resample(n2_samples, replace=False,
                                  n_samples=num_samps, random_state=0)

        return cdist(self.thedata[n1_samples], self.thedata[n2_samples], 'euclidean').min()

    def exp_maxdist(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]
        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        # If sample > 500 do sampling
        if len(n1_samples) > num_samps:
            n1_samples = resample(n1_samples, replace=False,
                                  n_samples=num_samps, random_state=0)
        if len(n2_samples) > num_samps:
            n2_samples = resample(n2_samples, replace=False,
                                  n_samples=num_samps, random_state=0)

        return cdist(self.thedata[n1_samples], self.thedata[n2_samples], 'euclidean').max()

    def exp_avgdist(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]
        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        # If sample > 500 do sampling
        if len(n1_samples) > num_samps:
            n1_samples = resample(n1_samples, replace=False,
                                  n_samples=num_samps, random_state=0)
        if len(n2_samples) > num_samps:
            n2_samples = resample(n2_samples, replace=False,
                                  n_samples=num_samps, random_state=0)
        return cdist(self.thedata[n1_samples], self.thedata[n2_samples], 'euclidean').mean()

    def exp_axis_min_dist(self, n1, n2):

        if(n1 == n2):
            return float('inf')

        n1_samples = self.node_to_exp[n1[0]]
        n2_samples = self.node_to_exp[n2[0]]
        num_samps = np.min([len(n1_samples), len(n2_samples), 500])

        # If sample > 500 do sampling
        cd = cdist(self.thedata[n1_samples],
                   self.thedata[n2_samples], 'euclidean')
        min0 = np.min(cd, axis=0).mean()
        min1 = np.min(cd, axis=1).mean()
        return (min0 + min1) / 2
# Class begin


class ELHierarchy(Baseclustering):
    """Class of of the clustering algorithms describe in the elife paper.

    Parameters
    ----------
    data_size: integer, optional, defalut: 100
        Choose subset size of the input data. Since the compute can be slow on full dataset,
        modify data_size to run less rounds

    metric: string, optional, default: 'node_distance'
        Disntance metric between nodes. One common version and one boosting version


    Attributes
    ----------
    metric_dic: dictionary
        A dictionary thta parse metric string to callable function

    thedata: array-like
        The input training dataset.

    thindices: array-like
        Experiment index in the whole dataset

    exp_to_range: dictionary
        Parse a 2-d tuple to its corresponding experiment index.
        e.g. (35,33) can get a set of experiment index [1,2,3,4,45]

    node_to_line: dictionary
        Parse a node ailas (represent by a 1-d vector) to a set of feature vectors

    node_to_exp: dictionary
        Parse a node alias (represent by a 1-d vecotr) to a set tuple of experiemnt condition,target pair

    nodes: array-like
        Encoded node alias, each node correspond to a set of feature vectors

    Z: array-like
        A linkage information in the format in scipy, see scipy.cluster.hierarchy.linkage

    labels_: array-like
        Clustering label of the input data


    """

    # def __init__(self):
    #     self.metric = 'node_1nn'
    #     self.metric_dic = {
    #         'node_1nn': self.exp_dis,
    #         'boost_node_1nn': self.exp_dis_boost,
    #         'single_link': self.exp_cdist,
    #         'group_average': self.exp_avgdist,
    #         'complete_link': self.exp_maxdist,
    #     }

    def fit(self, data, index, link='my_merge', metric='node_1nn'):
        """ Input data to generate cluserting result

        Parameters:
        ----------
        data: array-like, shape(n_query, n_features), 
            Test samples.

        index: array-like, shape(n_query, 2), 
            Experiemnt labels of a test sample, each tuple contains [target,condiction]

        n_sample: integer, number of sample,
            Sample a subset of the orignal dataset (no shuffle, just for unit test)

        link: string, _, deafult = 'my_merge'
            Linkage method to use in generating Z, possible values referer to scipy.cluster.hierarchy.linkage

        metric: string, _, default = 'single_link'
            Type of distances between to nodes, all possible values are in the attribute metric_dic

        my_merge: boolean, _, default: False
            To use the merging stragedy: the two closest node pair will become one, then the corresponding node data are union of the two nodes.
            Merge end until it becoome a singleton cluster.

        """
        self.metric = metric
        self.thedata = data
        self.theindices = index
        self.exp_to_range = dict()

        # Build a index that inflect experiment to expeirment indecies
        for i in range(self.theindices.shape[0]):
            if (self.theindices[i, 0], self.theindices[i, 1]) not in self.exp_to_range:
                self.exp_to_range[(self.theindices[i, 0],
                                   self.theindices[i, 1])] = []
            self.exp_to_range[(self.theindices[i, 0],
                               self.theindices[i, 1])].append(i)

        idx = np.array(list(range(0, len(self.exp_to_range))))
        self.node_to_line = dict(zip(idx, self.exp_to_range.keys()))
        self.node_to_exp = dict(zip(idx, self.exp_to_range.values()))
        self.nodes = idx.reshape(-1, 1)
        self.dmat = np.zeros(shape=(len(self.nodes), len(self.nodes)))

        if(link != 'my_merge'):
            self.Z = linkage(
                y=self.nodes, metric=self.metric_dic[self.metric], method=link)
            self.node_to_leaves, self.node_to_exp = linkage_to_leaves(
                Z=self.Z, data_size=len(self.nodes), node_to_exp=self.node_to_exp)
            self.dmat = squareform(
                pdist(self.nodes, metric=self.metric_dic[self.metric]))

        else:
            self.Z = self.my_merge(
                nodes=self.nodes, metric=self.metric_dic[self.metric])
            self.node_to_leaves, n2exp_placeholder = linkage_to_leaves(
                Z=self.Z, data_size=len(self.nodes), node_to_exp=self.node_to_exp)

            # Edit the node size to be consistence to the linkage generated in the scipy
            node_size = [len(item) for item in self.node_to_leaves.values()]
            self.Z[:, 3] = node_size

    def my_merge(self, nodes, metric):
        """ My own method to generate the clustering tree with record of each step statistics

        Paramerers:
        ----------
        nodes: array-like, shape(number of samples, 1), vector
            A 1-d vector that index a set of experiments

        metric: string
            Type of distances between to nodes, all possible values are in the attribute metric_dic


        Return:
        ------
        linkage:array-like
            A linkage information in the format in scipy, see scipy.cluster.hierarchy.linkage


        """
        node_queue = nodes
        linkages = np.array([])
        dmat = np.array([])
        merge = tuple()

        while(node_queue.shape[0] > 1):

            if(dmat.size < 1):
                # Create distance matrix for all nodes in each round
                result = pdist(node_queue, metric=metric)
                dmat = squareform(result)
                self.dmat = np.array(dmat)
                np.fill_diagonal(dmat, float('inf'))

            else:
                c_node = node_queue.max()
                dis_row = cdist(node_queue, [[c_node]], metric=metric)
                dmat = np.append(dmat, dis_row[:-1].reshape(-1, 1), axis=1)
                dmat = np.append(dmat, dis_row.reshape(1, -1), axis=0)
                np.fill_diagonal(dmat, float('inf'))

            # Select the nearest pari of nodes
            merge_idx = list(nd_argmin(dmat))
            merge = node_queue[merge_idx].ravel()

            num_sam = len(self.node_to_exp[merge[0]]) + \
                len(self.node_to_exp[merge[1]])

            # Fill merge pairs in each round
            linrow = np.array([merge[0], merge[1], dmat.min(), num_sam])

            # Record new node information
            newnode = list(self.node_to_exp[merge[0]])
            newnid = node_queue.max() + 1
            newnode.extend(self.node_to_exp[merge[1]])
            self.node_to_exp[newnid] = newnode

            # Pop merged nodes in the queue, insert new node to the queue
            node_queue = np.delete(node_queue, merge_idx, axis=0)
            node_queue = np.append(node_queue, [[newnid]], axis=0)

            # Pop merged nodes in the matrix
            dmat = np.delete(dmat, merge_idx, axis=0)
            dmat = np.delete(dmat, merge_idx, axis=1)

            # Append node to linkage info
            if(len(linkages.ravel()) < 4):
                linkages = linrow

            else:
                linkages = np.append(linkages, linrow, axis=0)
        return linkages.reshape(-1, 4)

    def predict(self, t, criterion='maxclust', depth=2, R=None, monocrit=None, splits=1):
        """ Return the class labels for the provide data

        Parameters:
        ----------
        t: integer or double or any kinds of number
            The threshold that prune the tree with a given criterion

        criterion: string
            The criteiron type that prune the tree, see in scipy.cluster.hierarchy.flcluster

        depth: integer
            See in scipy.cluster.hierarchy.flcluster

        R: 
            see in scipy.cluster.hierarchy.flcluster

        monocrit:
            see in scipy.cluster.hierarchy.flcluster


        Returns:
        -------
        self.labels_: array-like
            The clusering label of the input data samples

        """
        if(criterion == 'split'):

            split = []
            self.split_to_node = dict()
            no_list = self.nodes.ravel()

            for item in no_list:
                exps = self.node_to_exp[item]

        self.labels_ = fcluster(
            Z=self.Z, t=t, criterion=criterion, depth=depth, R=R, monocrit=monocrit)

        return self.labels_


# Class begin, use sepctual to generate clustering
class ELSpectual(Baseclustering):

    def fit(self, data, index, metric='node_1nn'):

        self.metric = metric
        self.thedata = data
        self.theindices = index
        self.exp_to_range = dict()

        # Build a index that inflect experiment to expeirment indecies
        for i in range(self.theindices.shape[0]):
            if (self.theindices[i, 0], self.theindices[i, 1]) not in self.exp_to_range:
                self.exp_to_range[(self.theindices[i, 0],
                                   self.theindices[i, 1])] = []
            self.exp_to_range[(self.theindices[i, 0],
                               self.theindices[i, 1])].append(i)

        idx = np.array(list(range(0, len(self.exp_to_range))))
        self.node_to_line = dict(zip(idx, self.exp_to_range.keys()))
        self.node_to_exp = dict(zip(idx, self.exp_to_range.values()))
        self.nodes = idx.reshape(-1, 1)
        self.dmat = squareform(
            pdist(self.nodes, metric=self.metric_dic[self.metric]))

        return

    def predict(self, n_clusters=8, eigen_solver='arpack', random_state=0, n_init=10, eigen_tol=0.0001, assign_labels='kmeans', n_jobs=1, delta=0.5):

        spectual = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver,
                                      random_state=random_state, n_init=n_init, eigen_tol=eigen_tol, assign_labels=assign_labels, affinity='precomputed', n_jobs=-1)

        self.affinity_matrix_ = np.exp(- self.dmat ** 2 / (2. * delta ** 2))

        spectual.fit(self.affinity_matrix_)

        return spectual.labels_ + 1


# Class begin, use label from the elife paper to do generate predictions
class SupervisedEL:
    """Class of of the supervised algorithms to determine clustering label. 
        Because the elife paper does not provide code, but provides labels for each round.
        If we choose some experiments that are not supposed to be tested in that round by the elife paper. 
        We would use a supervised model to predict their clustering labels.
        The training data is observed data proposed by the elife paper. 
        the training labels are the clustering labels in the intermediate result of the elife paper.

    Parameters
    ----------
    classifier: A sklearn classifier
        A classfier to be trained and make predictions

    label_encoder: A sklearn label encoder
        To preprocess the labler encoding.

    query_df: A pandas dataframe, must have columns: ['condition','targets']
        A set of conditions and targets queries that supposed to be observed in this round under the critira of my active learning algorithm.
    
    el_roundap: A pandas dataframe, must have columns: ['condition','targets','phenotype1','phenotype2','observed','frointer']
        A set of conditions and targets queries that supposed to be observed in this round under the critira of elfie paper.
        It is an intermediate result of their active learning proecess stored in the apredction files.
    

    Attributes
    ----------
    nmzsc_df: A pandas dataframe, must have columns: ['condition','targets', SLF features...]
        Data features with condition and target labels

    clf: A sklearn classifier
        A classfier to be trained and make predictions

    le: A sklearn label encoder
        To preprocess the labler encoding.

    query_df: A pandas dataframe, must have columns: ['condition','targets']
        A set of conditions and targets queries that supposed to be observed in this round under the critira of my active learning algorithm.
    
    elround_df: A pandas dataframe, must have columns: ['condition','targets','phenotype1','phenotype2','observed','frointer']
        A set of conditions and targets queries that supposed to be observed in this round under the critira of elfie paper.
        It is an intermediate result of their active learning proecess stored in the apredction files.

    """

    def __init__(self, classifier, label_encoder, query_df, el_roundap):
        self.clf = classifier
        self.elround_df = el_roundap
        self.query_df = query_df
        self.le = label_encoder
    
    def fit_predict(self, data):
        """ Funtions to fit the data into the classifier

        Parameter:
        ---------
        data:  A pandas dataframe, must have columns: ['condition','targets', SLF features...]
        Data features with condition and target labels


        Return:
        ------
        exped_lines: np.ndarray, shape = (n,2)
            Experiment combinations that have been observed till this round

        labels: np.ndarray, shape = (n,) 
            the predction labels for each experiment combinations (condition,target)
        """
        nmzsc_df = data

        # Split into observed part and unobserved part
        in_elround_exps = self.elround_df.loc[self.elround_df.observed ==1]
        out_elround_exps = self.elround_df.loc[self.elround_df.observed ==0]

        # The intersection of our selections and elife selections
        exped_in_el_df  = self.query_df.merge(in_elround_exps)
        exped_out_el_df = self.query_df.merge(out_elround_exps)

        # The data with feature split with observation labels
        in_el_data_df  = nmzsc_df.merge(exped_in_el_df)
        out_el_data_df  = nmzsc_df.merge(exped_out_el_df)

        logging.info("{0} experiments match elife, {1} experiments are predtions".format(len(exped_in_el_df),len(exped_out_el_df)))

        # Train a clf using the elife labels with KNN model
        X_train,y_train = in_el_data_df.loc[:,'ft_0':'ft_113'].as_matrix(),self.le.fit_transform(in_el_data_df.phenotype1.as_matrix())
        self.clf.fit(X=X_train,y=y_train)

        # The predictions are made upon data that have not been oberved by the elife paper
        X_unlabeled = out_el_data_df.loc[:,'ft_0':'ft_113'].as_matrix()
        y_predictions = self.clf.predict(X_unlabeled)

        # Inverse transfrom the labels to the original annotations
        out_el_predictions = self.le.inverse_transform(y_predictions)

        # Inferer prediction using the pularity label inside each group
        out_el_data_df['phenotypep'] = out_el_predictions
        out_el_pheno_df = out_el_data_df.groupby(['condition','target']).phenotypep.agg(lambda x: x.value_counts().idxmax()).to_frame().reset_index().astype(int)

        # Look up the labeled tuples in the elife aprediction intermidiate data
        in_el_pheno_df = exped_in_el_df.iloc[:,:3].rename(index=str,columns={'phenotype1':'phenotypep'})

        # Stack predition and lookups
        data_p_df = pandas.concat([in_el_pheno_df,out_el_pheno_df],ignore_index =True)
        
        labels = data_p_df.phenotypep.as_matrix()
        exped_lines = data_p_df.iloc[:,:2].as_matrix()

        return exped_lines, labels

def nd_argmin(A):
    """ Funtions to get the index of the smallest element in a 2-d matrix

    Parameter:
    ---------
    A: array-like, shape(x,2), number
    An arbitary 2-d matrix


    Return:
    ------
    np.unravel_index(A.argmin(), A.shape): tuple, (row, column), (number, number)
    A tuple of 2-d index of the smallest element in the matrix

    """
    return np.unravel_index(A.argmin(), A.shape)


def linkage_to_leaves(Z, data_size, node_to_exp):
    """ Convert the linkage formated matrix to a dictionary, 
        key is a node number while values is a array of leaves of the node

    Parameters:
    ----------
    Z: 2-d numpy matrix, shape(n_merges, 4), float
        The linkage matrix that generated by See in scipy.cluster.hierarchy.linkage

    data_size: integer, _, _
        The size of the orignal dataset, the same as the total number of leaves in the likage tree (dendrogram)


    Return:
    ------
    node_to_leaves: dictionary, (node number, set of leaves), (integer, list(integer))
        A dictionary that can parse a integer node (indexed by a integer) in the linkage tree (dendrogram), to it corresponding leaf nodes.

    """
    node_id = data_size
    node_to_leaves = {}

    for merge in Z[:, :2].astype(int):

        # Both node have node id < number of node, the two merging nodes are leaves
        if(merge.max() < data_size):
            node_to_leaves[node_id] = []
            node_to_leaves[node_id].extend(merge)

        # One of the merging node (the one with small node index) is a leaf node
        elif(merge.min() < data_size):

            s, l = merge.min(), merge.max()
            node_to_leaves[node_id] = []
            node_to_leaves[node_id].extend(node_to_leaves[l])
            node_to_leaves[node_id].append(s)

        # Both nodes are internal nodes
        else:
            node_to_leaves[node_id] = []
            node_to_leaves[node_id].extend(node_to_leaves[merge[0]])
            node_to_leaves[node_id].extend(node_to_leaves[merge[1]])

        # Merge the
        new_node_exp = list(node_to_exp[merge[0]])
        new_node_exp.extend(node_to_exp[merge[1]])
        node_to_exp[node_id] = new_node_exp

        node_id = node_id + 1

    return node_to_leaves, node_to_exp

# Fast sampling


def fast_sampling(series, num, seed=0):
    random.Random(seed).shuffle(series)
    return series[:num]
