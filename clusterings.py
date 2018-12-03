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
    
    elround_ap: A pandas dataframe, must have columns: ['condition','targets','phenotype1','phenotype2','observed','frointer']
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
