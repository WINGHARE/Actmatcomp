
import os
import string
import sys
import logging
from optparse import OptionParser

import numpy as np
import pandas
import scipy.spatial.distance
import scipy.stats
from numpy.linalg import matrix_rank
from scipy.spatial import distance
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier

import alearn_utils as utils
import clusterings as elCluster
import mf_impute as mfi
import fancy_impute as fci


# Parameter parsing
#############################################################################################################
argvs = sys.argv
opts, args = {}, []
op = OptionParser()
op.add_option(
    '-c',
    '--cluster',
    action='store',
    type='string',
    dest='cluster',
    help='indicate the ID of the job')
op.add_option(
    '-d',
    '--date',
    action='store',
    type='string',
    dest='date',
    help='indicate the time of the job')
op.add_option(
    '-m',
    '--metric',
    default='node_1nn',
    action='store',
    type='string',
    dest='metric',
    help='metric of clustering, if not clutsering is used it is dummy')
op.add_option(
    '-l',
    '--linkage',
    default='predictround',
    action='store',
    type='string',
    dest='linkage',
    help='linkage of clustering, since no clustering is using, we use predictions')
op.add_option(
    '-b',
    '--batchsize',
    default=-1,
    action='store',
    type='int',
    dest='batchsize',
    help='batchsize of clustering metric, if -1, the number is pre-determinined')
op.add_option(
    '-o',
    '--outputpath',
    default='apreds',
    action='store',
    type='string',
    dest='outputpath',
    help='Output path, please specify one before running')
op.add_option(
    '-p',
    '--impute',
    default='svt',
    action='store',
    type='string',
    dest='impute',
    help='Impute method, the matrix completioin method')
op.add_option(
    '-n',
    '--condition',
    default=96,
    action='store',
    type='int',
    dest='condition',
    help='Condition size of the problem,')
op.add_option(
    '-t',
    '--target',
    default=96,
    action='store',
    type='int',
    dest='target',
    help='Target size of the problem,')
op.add_option(
    '-r',
    '--rounds',
    default=30,
    action='store',
    type='int',
    dest='rounds',
    help='Iterations of the learning procedure')
op.add_option(
    '-y',
    '--phenotypes',
    default=-1,
    action='store',
    type='int',
    dest='phenotypes',
    help='Number of max phenotypes')
op.add_option(
    '-u',
    '--duplicate',
    default=0,
    action='store',
    type='int',
    dest='duplicate',
    help='Duplicate of the experiment space, 1: duplicate, 0: reserve 3200')
op.add_option(
    '-s',
    '--select',
    default='margin',
    action='store',
    type='string',
    dest='select',
    help='Selection querying method of active learning')
op.add_option(
    '-f',
    '--info',
    default='NA',
    action='store',
    type='string',
    dest='info',
    help='Information to print in the heading to record some important marks')

(opts, args) = op.parse_args()

if len(args) > 0:
    op.print_help()
    op.error('Please input options instead of arguments.')
    exit(1)

NUM_TAR = opts.condition
NUM_CON = opts.target
BATCHSIZE = opts.batchsize
DUPLICATE = opts.duplicate
ROUNDS = opts.rounds
METRIC = opts.metric
IMPUTE = opts.impute
LINKAGE = opts.linkage
OPATH = opts.outputpath
SELECT = opts.select
NUM_PHE = opts.phenotypes

logging.basicConfig(level=logging.INFO)
logging.info(opts)

# Parameter parsing ends
#############################################################################################################


# Data preprocessing
#############################################################################################################
data_idx = np.load('data/calculated/indices.npy')
#raw_data = np.load('data/calculated/rawdata.npy')
zsc_data = np.load('data/calculated/zscored_data.npy')
#pdata_idx = np.load('data/calculated/indices_posthoc.npy')
#pzsc_data = np.load('data/calculated/zscored_data_posthoc.npy')

# Determine number of batch and pheno types
batchsize_list = np.full(ROUNDS, BATCHSIZE)
phenonum_list = np.full(ROUNDS, NUM_PHE)
if (BATCHSIZE == -1):
    batchsize_list = np.load('batchsize.npy')
if (NUM_PHE == -1):
    phenonum_list = np.load('phenotypenums.npy')

# If smaller space
if(NUM_TAR == 48 and NUM_CON == 48):
    data_idx = (data_idx - 1) % 48 + 1

# Generate data frame
zdata_with_idx = np.hstack((data_idx, zsc_data))
zdata_with_idx[:, :2]
col_name = ['condition', 'target']
featurename = ['ft_' + str(i) for i in range(zsc_data.shape[1])]
col_name.extend(featurename)
zdata_df = pandas.DataFrame(zdata_with_idx, columns=col_name)
zdata_df = zdata_df[(zdata_df.target <= NUM_TAR) &
                    (zdata_df.condition <= NUM_CON)]

# Get avaliable space
full_set = set()
for c in range(1, NUM_CON + 1):
    for t in range(1, NUM_TAR + 1):
        full_set.add((c, t))


ava_exps = utils.get_exp_from_df(zdata_df)
na_exps = full_set - ava_exps
logging.info("unavaliable experiments: {0}".format(len(na_exps)))
logging.info("avaliable experiments: {0}".format(
    len(utils.get_exp_from_df(zdata_df))))

# Mapping true for duplicated usage.
exp_map = {}
for item in ava_exps:
    c, t = (item[0] - 1) % 48 + 1, (item[1] - 1) % 48 + 1
    exp_map[(c, t)] = (item[0], item[1])

nmzsc_data = zsc_data
nmzsc_idx = np.hstack((data_idx, nmzsc_data))
nmzsc_df = pandas.DataFrame(nmzsc_idx, columns=col_name)
nmzsc_df = nmzsc_df[(nmzsc_df.target <= NUM_TAR) &
                    (nmzsc_df.condition <= NUM_CON)]


# Data preprocessing ends
#############################################################################################################


# Initial round to select the control group
#############################################################################################################

# Get true experimnets form pool
exped_df = nmzsc_df[(nmzsc_df.condition == 48)]
exped_set = utils.get_exp_from_df(exped_df)

# Add in 0906 predict round method that build precition using supervised model in each round
if(LINKAGE == 'predictround'):
    ap_df = pandas.read_csv("data/elap/bear_round" +
                            str(1) + ".apredictions", sep=" ", header=None)
    ap_df.columns = ['condition', 'target', 'phenotype1',
                     'phenotype2', 'observed', 'frontier']
    ap_df = ap_df.add([0, 0, 1, 1, 0, 0], axis=1)
    ap_ob1_df = pandas.DataFrame(ap_df.loc[ap_df.observed == 1])
    con48_r1_phe = ap_ob1_df.loc[ap_ob1_df.condition == 48]
    exped_lines = con48_r1_phe.iloc[:, :2].as_matrix()
    ini_exped_labels = con48_r1_phe.phenotype1.as_matrix()

exp_pheno_df = pandas.DataFrame(
    exped_lines, columns=[['condition', 'target']])
exp_pheno_df['phenotype'] = ini_exped_labels
data_df = exp_pheno_df.astype(int)

# Initial predictive model
for item in full_set - exped_set:
    tup = exp_pheno_df[exp_pheno_df.target == item[1]].as_matrix()
    tup[0, 0] = item[0]
    new_row = pandas.DataFrame(
        tup, columns=['condition', 'target', 'phenotype'])
    data_df = data_df.append(new_row)

if(DUPLICATE == True):
    ini_imp = utils.gen_first_batch(full_set - exped_set, batchsize_list[1])
else:
    ini_imp = utils.gen_first_batch(ava_exps - exped_set, batchsize_list[1])

# Initial report
apred_1 = utils.get_aprediction(data_df, data_df.phenotype, exped_set, ini_imp)
apred_1.to_csv(OPATH + '/bear_round' + str(1) +
               '.apredictions', header=False, index=False, sep=' ')

# Initial round ends
#############################################################################################################

# Active learning processing methods
#############################################################################################################


def update_imputed_frame(data_df, exped_df, exped_set, impute_set, which_round, method='three_point', duplicate=False):

    missing_flag = 0
    # Load data from the pool
    for item in (impute_set):

        condition = item[0]
        target = item[1]
        new_exps = nmzsc_df[(nmzsc_df.condition == condition)
                            & (nmzsc_df.target == target)]
        # Append true experiment to the data matrix check avaliability of experiments

        # Experiment avaliable
        if(len(new_exps) > 0):
            exped_df = exped_df.append(new_exps)
            exped_set.add(item)
        #
        elif ((duplicate == True) and (item in na_exps)):
            missing_flag += 1
            d_condition, d_target = exp_map[(
                (condition - 1) % 48 + 1, (target - 1) % 48 + 1)]
            new_exps = pandas.DataFrame(
                nmzsc_df[(nmzsc_df.condition == d_condition) & (nmzsc_df.target == d_target)])
            new_exps.condition = np.full(len(new_exps), condition)
            new_exps.target = np.full(len(new_exps), target)
            exped_df = exped_df.append(new_exps)
            exped_set.add(item)

    logging.warning(
        "{0} expriment is not avaliable in this batch".format(missing_flag))

    # To study the data from their files
    if(LINKAGE == 'predictround'):

        # The experimenrs we want to perform
        query_ary = np.array(list(exped_set))
        query_df = pandas.DataFrame(
            query_ary, columns=[['condition', 'target']])

        # Load the active prediction from the elife paper
        elround_df = pandas.read_csv(
            'data/elap/bear_round' + str(which_round + 1) + '.apredictions', sep=' ', header=None)
        elround_df = elround_df.add([0, 0, 1, 1, 0, 0], axis=1)
        elround_df.columns = ['condition', 'target',
                              'phenotype1', 'phenotype2', 'observed', 'frontier']

        # Construct classifier and the label encoder
        le = LabelEncoder()
        knC = KNeighborsClassifier(
            n_neighbors=15, metric="euclidean", weights='distance', n_jobs=8)

        clustering = elCluster.SupervisedEL(
            classifier=knC, label_encoder=le, query_df=query_df, el_roundap=elround_df)
        exped_lines, labels = clustering.fit_predict(data=nmzsc_df)

    # Add in 0906

    logging.warning(
        "{0} number of clusters are generated".format(np.max(labels)))

    #exped_lines = np.array(list(clustering.node_to_line.values()))
    data_df = pandas.DataFrame(exped_lines, columns=[['condition', 'target']])
    data_df['phenotype'] = labels

    var_mat = np.full((NUM_TAR, NUM_CON), -1)

    if(method == 'svt'):

        # Generate the phenotype matrix and phenotype list
        phenotype_list = np.array(data_df.drop_duplicates(
            subset=['phenotype']).phenotype.as_matrix(), dtype=int)
        phenotype_matrix = np.full((NUM_CON, NUM_TAR), 0)

        for item in exped_set:
            c, t = item[0], item[1]
            phe_for_ct = int(
                data_df.loc[(data_df.condition == c) & (data_df.target == t)].phenotype)
            phenotype_matrix[t - 1, c - 1] = phe_for_ct

        if(method == 'svt'):
            prediction, score_stack, correct_count = fci.svt_impute(
                phenotype_matrix, phenotype_list, phenotype_matrix.shape, dummy_na=0.5, mat_rank=-1)

        logging.info("The training accurarcy of the matrix factorization is {0}".format(
            correct_count / len(exped_set)))

        var_mat = np.zeros((NUM_TAR, NUM_CON))
        # Generate the score using margin uncertanty
        if(SELECT == 'margin'):
            var_mat = mfi.min_margin_score(score_stack)

        elif (SELECT == 'least'):
            var_mat = mfi.least_confidence_score(score_stack)
            # Decline the experimened
        var_mat = var_mat - 2 * (phenotype_matrix > 0)

        if(duplicate == False):
            for item in na_exps:
                c, t = item[0], item[1]
                var_mat[t - 1, c - 1] = -1

            # Generate the prediction based on the imputed matrix
        c_t = np.array(np.unravel_index(
            list(range(0, NUM_CON * NUM_TAR)), (NUM_CON, NUM_TAR))) + 1
        pheno_pred_list = prediction.T.ravel()
        data_array = np.append(
            c_t.T, pheno_pred_list.reshape(-1, 1), axis=1)
        data_df = pandas.DataFrame(
            data_array, columns=['condition', 'target', 'phenotype'])

    return data_df, exped_df, exped_set, var_mat


def per_round_impure_labels(data_df, exped_df, exped_set, imp_set, which_round):

    data_df, exped_df, exped_set, var_mat = update_imputed_frame(
        data_df, exped_df, exped_set, imp_set, which_round, method=IMPUTE, duplicate=DUPLICATE)

    return data_df, exped_df, exped_set, var_mat

# Active learning processing methods ends
#############################################################################################################


# Active learning processing rounds 2 - finish
#############################################################################################################
labels = np.array([])
frointer_set = ini_imp
for i in range(1, ROUNDS):

    logging.info("Round {0}".format(i + 1))

    data_df, exped_df, exped_set, var_matrix_next = per_round_impure_labels(
        data_df, exped_df, exped_set, frointer_set, i)

    logging.info("{0} experiment is observed".format(len(exped_set)))

    batch = batchsize_list[i] if i < 30 else 96

    logging.info("{0} experiment is selected as frointer".format(batch))

    if((var_matrix_next >= 0).sum() >= batch):
        logging.info('Use active learning selection')
        frointer_set = utils.selection(var_matrix_next, batch)
        logging.info('The higest score: {0}'.format(var_matrix_next.max()))

    elif (i + 1) < 30:
        logging.info(
            'Use active learning selecion is not enough, use random sampling')
        frointer_set = utils.random_gen(
            ava_exps - exped_set, batch_size=batch)

    else:
        logging.info('Final round we dont have enough data so random selections')
        frointer_set = utils.random_gen(full_set - exped_set, batch_size=batch)

    apred_df = utils.get_aprediction(
        data_df, data_df.phenotype, exped_set, frointer_set)

    apred_df.to_csv(OPATH + '/bear_round' + str(i + 1) +
                    '.apredictions', header=False, index=False, sep=' ')
# Active learning processing rounds 2 - ends
#############################################################################################################
