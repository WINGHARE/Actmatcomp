import string
import scipy.stats
import pandas
import numpy as np
import secrets
from random import shuffle
from sklearn.utils import resample


def get_exp_from_df(data_frame):
    cons = data_frame.drop_duplicates(subset=['condition', 'target']).as_matrix()[
        :, 0].astype(int)
    tars = data_frame.drop_duplicates(subset=['condition', 'target']).as_matrix()[
        :, 1].astype(int)
    exps = set(zip(cons, tars))
    return exps


def entropy(data):
    p_data = np.bincount(data) / len(data)
    # input probabilities to get the entropy
    entropy = scipy.stats.entropy(p_data[np.nonzero(p_data)[0]], base=2)
    return entropy

# Pick up K maximum number from a 2d matrix


def top_k_idx(a, k, descend=True):
    if (descend == False):
        return np.argsort(a)[:k]
    return np.argsort(a)[::-1][:k]

# Rand gen


def random_gen(ava_set, batch_size=96):
    ava_list = list(ava_set)
    samples_list = resample(ava_list, n_samples=batch_size,
                            replace=False, random_state=0)
    return set(samples_list)

# Exp in set


def exp_in_set(row, the_set):
    return int((row[0], row[1]) in the_set)

# Get apred


def get_aprediction(data_df, labels, exped_set, imp_set):
    exp_pheno = data_df.iloc[:, :2].astype(int)
    exp_pheno['phenotype'] = labels.astype(int)
    exp_pheno['phenotype2'] = labels.astype(int)
    exp_pheno.sort_values(by=['condition', 'target'],
                          ascending=True, inplace=True)
    exp_pheno['observed'] = exp_pheno.apply(
        lambda x: exp_in_set(x, exped_set), axis=1)
    exp_pheno['frointer'] = exp_pheno.apply(
        lambda x: exp_in_set(x, imp_set), axis=1)

    return exp_pheno

# Max selection


def selection(score_matrix, batchsize, descend=True):

    batch_idx = top_k_idx(score_matrix.ravel(), batchsize, descend=descend)
    num_tar, num_con = score_matrix.shape
    con = (batch_idx % num_tar) + 1
    tar = (np.floor(batch_idx / num_tar).astype(int)) + 1

    return set(zip(con, tar))

# Soft max by axis


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Generate first batch


def gen_first_batch_full(batchsize):
    """Generate the initial imputation"""
    conditions = list(range(1, 97))
    conditions.remove(48)
    conditions.append(secrets.choice(conditions))
    shuffle(conditions)
    targets = list(range(1, 97))
    shuffle(targets)
    sets = set(zip(conditions[:batchsize], targets[:batchsize]))
    return sets


def gen_first_batch(ava_exps, batchsize):
    """Generate the initial imputation"""
    ava_list = [[i[0], i[1]] for i in ava_exps]
    shuffle(ava_list)

    dict_a = {}

    for item in ava_list:
        c, t = item[0], item[1]
        if((c not in dict_a) and t <= 96):
            dict_a[c] = [c, t]

    frointers = np.array(list(dict_a.values()))

    rest = len(frointers) - batchsize

    if(rest >= 0):
        return set(zip(frointers[:batchsize, 0], frointers[:batchsize, 1]))
    else:
        resultset = set(zip(frointers[:, 0], frointers[:, 1]))
        ava = ava_exps - resultset
        for i in range(0, rest):
            resultset.add(ava.pop())
    return resultset


class ALMatSimulator:
    """ Funtion to generate the simulation matrix of to test the performance of active machine learning problem

    Parameter:
    ---------
    shape:  A tuple, (targets,conditions)
        Data features with condition and target labels

    num_pheno: integer, number of different phenotypes
        Number of uniques labels in the matrix

    response: float,
        The average percentage that any target can preserve its label across different conditions

    unique: float,
        The percentage of the unique phenotype labels combinations of target-condition pais.


    Attributes:
    ------
    shape:  A tuple, (targets,conditions)
        Data features with condition and target labels

    num_pheno: integer, number of different phenotypes
        Number of uniques labels in the matrix

    response: float,
        The average percentage that any target can preserve its label across different conditions

    unique: float,
        The percentage of the unique phenotype labels combinations of target-condition pais.

    pheno_full: np.ndarray, shape = (target,condition)
        A simulation matrix of target-condition matrix. Values are phenotype labels.
        The first column is the unperturbed (control group, condition[0]) targets
            
    control_group: np.ndarray, shape = (target,1)
        The unperturbed control group of the expriement
        
    perturb_count: integer
        The perturbation count of unique targets

    """

    def __init__(self, shape, num_pheno=8, response=0.5, unique=0.5):
        self.shape = shape
        self.num_pheno = num_pheno
        self.response = response
        self.unique = unique

    def gen_simulation_mat(self, random_state=0, reserve=True):
        """ Funtion to generate the simulation matrix of to test the performance of active machine learning problem

        Parameter:
        ------
        random_state: integer,
            The random seed to generate random sample

        reserve: boolean,
            The control flag to control random sampling, if True, then all unique samples are reserved.


        Return:
        ------
        pheno_full: np.ndarray, shape = (target,condition)
            A simulation matrix of target-condition matrix. Values are phenotype labels.
            The first column is the unperturbed (control group, condition[0]) targets


        """
        # initialize parameters
        condition, target = self.shape[0], self.shape[1]
        rstate = np.random.RandomState(random_state)

        # generate initialized ary of unperturbed experiemnts
        vie_only_unique_ary = rstate.randint(low=1, high=(
            self.num_pheno + 1), size=int(target * self.unique))

        # Copy the viechal only (control group) wating for pertuabation
        unique_phenos_tars = resample(vie_only_unique_ary.reshape(
            1, -1), n_samples=int(condition * self.unique), random_state=random_state)

        # The mask that perturb the control groupu phenotype
        perturbe_mask = rstate.choice([1, 0], size=unique_phenos_tars.shape, p=[
                                      self.response, 1 - self.response])
        np.put(perturbe_mask, list(range(0, perturbe_mask.shape[0])), [0])

        # Perturbed values to add, add a values to the original matrix and mod it so it wont be the original value
        add = rstate.randint(low=1, high=self.num_pheno +
                             1, size=unique_phenos_tars.shape)
        perturbed_tars = ((add * perturbe_mask) +
                          unique_phenos_tars) % (self.num_pheno + 1)
        
        # The 0 values means it is added by a value that orgin+add == numtar +1, fill it then it is perturbed
        perturbed_tars = (perturbed_tars == 0) * add + perturbed_tars

        # Resample it to the number of the true targets, reserve the control group
        if(reserve == False):
            # Reserve == False, the maked up nuique samples are not garanteed to be reseverd
            pheno_tars = resample(
                perturbed_tars, n_samples=condition - 1, random_state=random_state, replace=True)
            pheno_tars = np.append(
                vie_only_unique_ary.reshape(1, -1), pheno_tars, axis=0)
            pheno_full = resample(
                pheno_tars.T, n_samples=target, random_state=random_state, replace=True)
        else:
            # Reserve == True, the maked up nuique samples are  garanteed to be reseverd
            pheno_tars_extra = resample(perturbed_tars, n_samples=condition - len(
                perturbed_tars), random_state=random_state, replace=True)
            pheno_tars = np.append(perturbed_tars, pheno_tars_extra, axis=0)
            pheno_full_extra = resample(pheno_tars.T, n_samples=target - len(
                pheno_tars.T), random_state=random_state, replace=True)
            pheno_full = np.append(pheno_tars.T, pheno_full_extra, axis=0)

        self.pheno_mat = pheno_full
        self.control_group = pheno_full[:,0]
        self.perturb_count = perturbe_mask.sum()

        return pheno_full
