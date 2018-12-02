import numpy as np
import logging
from numpy.linalg import matrix_rank
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA



## nmf_imputation
def nmf_impute(phenotype_matrix,phenotype_list,shape,dummy_na = 0.5,mat_rank = -1,Lambda = 1):
    """ The matrix complete based on nmf, the imputation is based on the one versus rest pattern.

    Parameters:
    ----------
    phenotype_matrix: np_array
        A numpy matrix that to be imputed
    
    phenotype_list: np array
        A list of all possible values in the matirx
    
    shape: tuple
        The shape of the impute matrix
    

    Returns:
    -------
    prediction:
        The imputed version of the matrix
    
    reconstruct_stacks:
        The low rank apporximation of each one v rest affiliation matrix

    correct_count:
        The how many element from the prediction is the same to the trainning matrix 

    """  
    
    exped_mask = (phenotype_matrix == 0).astype(int)
    logging.info("Thre rank of the done experiments in this round is {0}".format(matrix_rank(phenotype_matrix)))
 
    # Build one vs rest categorical imputation
    # Reconstruction stacks are score of each affliation matrix shape = 
    reconstruct_stacks = np.zeros(shape=(len(phenotype_list),shape[0],shape[1]))
    
    for i in range(0,len(phenotype_list)):
        each_phenotype_mat = (phenotype_matrix == phenotype_list[i]).astype(float)
        if (dummy_na != None):
            each_phenotype_mat += (dummy_na) * (exped_mask)
        
        rank = matrix_rank(each_phenotype_mat)
        # The the affiliation matrix for each experiemnt:
        # 0, 1 represents the affiliation (belongs to phenotype or not), 0.5 are the missing values.
    
        if( rank > 1):

            if mat_rank == -1:
                components = np.min([int(rank/2),rank])
            else:
                components = np.min([mat_rank,rank]) 
            nmf = NMF(n_components=components,alpha= Lambda)  #  Reuduce the rank of the matrix
            u_distribution = nmf.fit_transform(each_phenotype_mat)
            v_distribution = nmf.components_
            reconstruct_matrix = np.dot(u_distribution, v_distribution)
            reconstruct_stacks[i] = reconstruct_matrix.clip(0,1) #* exped_mask
        
        else :

            reconstruct_stacks[i] = each_phenotype_mat
    
    prediction = phenotype_list[reconstruct_stacks.argmax(axis=0)]
    
    predict_full = (prediction * exped_mask) + (phenotype_matrix *  np.logical_not(exped_mask))
    
    correct_count = ((phenotype_matrix == prediction) * np.logical_not(exped_mask)).sum()

    return predict_full, reconstruct_stacks, correct_count
## svd_imputation
def svd_impute(phenotype_matrix,phenotype_list,shape,dummy_na = 0.5,mat_rank = -1):
    """ The matrix complete based on nmf, the imputation is based on the one versus rest pattern.

    Parameters:
    ----------
    phenotype_matrix: np_array
        A numpy matrix that to be imputed
    
    phenotype_list: np array
        A list of all possible values in the matirx
    
    shape: tuple
        The shape of the impute matrix
    

    Returns:
    -------
    prediction:
        The imputed version of the matrix
    
    reconstruct_stacks:
        The low rank apporximation of each one v rest affiliation matrix

    correct_count:
        The how many element from the prediction is the same to the trainning matrix 

    """  
    
    exped_mask = (phenotype_matrix == 0).astype(int)
    logging.info("Thre rank of the done experiments in this round is {0}".format(matrix_rank(phenotype_matrix)))
 
    # Build one vs rest categorical imputation
    # Reconstruction stacks are score of each affliation matrix shape = 
    reconstruct_stacks = np.zeros(shape=(len(phenotype_list),shape[0],shape[1]))
    
    for i in range(0,len(phenotype_list)):
        each_phenotype_mat = (phenotype_matrix == phenotype_list[i]).astype(float)
        if (dummy_na != None):
            each_phenotype_mat += (dummy_na) * (exped_mask)
        
        rank = matrix_rank(each_phenotype_mat)
        # The the affiliation matrix for each experiemnt:
        # 0, 1 represents the affiliation (belongs to phenotype or not), 0.5 are the missing values.
    
        if( rank > 1):

            if mat_rank == -1:
                components = int(np.min([int(rank/2),rank]))
            else:
                components = int(np.min([mat_rank,rank])) 
            svd = TruncatedSVD(n_components=components)  #  Reuduce the rank of the matrix
            reconstruct_matrix = np.clip(svd.inverse_transform(svd.fit_transform(each_phenotype_mat)),0,1)
            reconstruct_stacks[i] = reconstruct_matrix.clip(0,1) #* exped_mask
        
        else :

            reconstruct_stacks[i] = each_phenotype_mat
    
    prediction = phenotype_list[reconstruct_stacks.argmax(axis=0)]
    
    predict_full = (prediction * exped_mask) + (phenotype_matrix *  np.logical_not(exped_mask))
    
    correct_count = ((phenotype_matrix == prediction) * np.logical_not(exped_mask)).sum()

    return predict_full, reconstruct_stacks, correct_count

def pca_impute(phenotype_matrix,phenotype_list,shape,dummy_na = 0.5,mat_rank = -1):
    """ The matrix complete based on nmf, the imputation is based on the one versus rest pattern.

    Parameters:
    ----------
    phenotype_matrix: np_array
        A numpy matrix that to be imputed
    
    phenotype_list: np array
        A list of all possible values in the matirx
    
    shape: tuple
        The shape of the impute matrix
    

    Returns:
    -------
    prediction:
        The imputed version of the matrix
    
    reconstruct_stacks:
        The low rank apporximation of each one v rest affiliation matrix

    correct_count:
        The how many element from the prediction is the same to the trainning matrix 

    """  
    
    exped_mask = (phenotype_matrix == 0).astype(int)
    logging.info("Thre rank of the done experiments in this round is {0}".format(matrix_rank(phenotype_matrix)))
 
    # Build one vs rest categorical imputation
    # Reconstruction stacks are score of each affliation matrix shape = 
    reconstruct_stacks = np.zeros(shape=(len(phenotype_list),shape[0],shape[1]))
    
    for i in range(0,len(phenotype_list)):
        each_phenotype_mat = (phenotype_matrix == phenotype_list[i]).astype(float)
        if (dummy_na != None):
            each_phenotype_mat += (dummy_na) * (exped_mask)
        
        rank = matrix_rank(each_phenotype_mat)
        # The the affiliation matrix for each experiemnt:
        # 0, 1 represents the affiliation (belongs to phenotype or not), 0.5 are the missing values.
    
        if( rank > 1):

            if mat_rank == -1:
                components = np.min([int(rank/2),rank])
            else:
                components = np.min([mat_rank,rank]) 
            pca = PCA(n_components=components) #  Reuduce the rank of the matrix
            reconstruct_matrix = np.clip(pca.inverse_transform(pca.fit_transform(each_phenotype_mat)),0,1)
            reconstruct_stacks[i] = reconstruct_matrix.clip(0,1) #* exped_mask
        
        else :

            reconstruct_stacks[i] = each_phenotype_mat
    
    prediction = phenotype_list[reconstruct_stacks.argmax(axis=0)]
    
    predict_full = (prediction * exped_mask) + (phenotype_matrix *  np.logical_not(exped_mask))
    
    correct_count = ((phenotype_matrix == prediction) * np.logical_not(exped_mask)).sum()

    return predict_full, reconstruct_stacks, correct_count

def min_margin_score(score_stack):

    sorted_prob = np.sort(score_stack, axis=0)
    var_mat = 1 - (sorted_prob[-1] - sorted_prob[-2])

    return  var_mat

def least_confidence_score(score_stack):

    confidence = score_stack.max(axis=0)
    return 1- confidence