import numpy as np

import numba
from numba import cuda

#from pyculib import rand


@numba.jit(nopython=True)
def permute_array_cpu(array, index_array):
    tmp_len = len(array)
    for i in range(tmp_len):
        index_array[i] = index_array[i] * tmp_len
        array[i], array[int(index_array[i])] = \
            array[int(index_array[i])], array[i]
            
@cuda.jit
def permute_array_gpu(array, index_array):
    tmp_len = len(array)
    for i in range(tmp_len):
        index_array[i] = index_array[i] * tmp_len
        array[i], array[int(index_array[i])] = \
            array[int(index_array[i])], array[i]


def create_permutation_matrix(y_bin_vars, n_reps=0):
    # Validate n_reps parameter
    assert n_reps >= 0, "n_reps value must be greater than or equal to 0!"
    
    if isinstance(y_bin_vars['bins'], np.ndarray):
        # execute on cpu
        
        # create a matrix that is going to hold the
        # original 'bins' series as well as all
        # the permutations. Flipping row and column
        # representation for easier execution
        mcpt_matrix = np.tile(y_bin_vars['bins'], (n_reps+1, 1))
        
        # copy 'bins' into first column of mcpt_matrix
        # and permute 'bins' n_reps number of times for
        # all other columns in mcpt_matrix
        for i in range(mcpt_matrix.shape[0]):
            if i > 0:
                rand_array = np.random.uniform(size=mcpt_matrix.shape[1])
                permute_array_cpu(mcpt_matrix[i], rand_array)
    else:
        # execute on gpu
        # copy y_bin_vars['bins'] back to the cpu in order to create
        # the mcpt_matrix
        bins = y_bin_vars['bins'].copy_to_host()
        # create the mcpt_matrix and copy back to gpu
        mcpt_matrix = np.tile(bins, (n_reps+1, 1))
        mcpt_matrix_gpu = cuda.to_device(mcpt_matrix)
        for i in range(mcpt_matrix_gpu.shape[0]):
            if i > 0:
                rand_array = \
                    rand.uniform(size=mcpt_matrix_gpu.shape[1], device=True)
                permute_array_gpu(mcpt_matrix_gpu[i], rand_array)
        mcpt_matrix = mcpt_matrix_gpu
        
    return mcpt_matrix


@numba.jit(nopython=True, parallel=True)
def solo_pvalue_calc(info_matrix, solo_p_val):
    for i in numba.prange(info_matrix.shape[0]):
        # mi_actual = info_matrix[i,0]
        solo_p_val[i] = 0
        for j in numba.prange(info_matrix.shape[1]):
            if j > 0:
                if info_matrix[i,0] < info_matrix[i,j]:
                    solo_p_val[i] += 1
        # Account for actual information value in pval calc
        solo_p_val[i] += 1
        solo_p_val[i] = solo_p_val[i]/(info_matrix.shape[1])
            

def solo_pvalue(info_matrix):
    solo_p_val = np.zeros(info_matrix.shape[0], dtype=np.float32)
    solo_pvalue_calc(info_matrix, solo_p_val)
    return solo_p_val


@numba.jit(nopython=True, parallel=True)
def _populate_max_info_array(info_matrix, max_info):
    for j in numba.prange(info_matrix.shape[1]):
        for i in range(info_matrix.shape[0]):
            if j > 0:
                if info_matrix[i,j] > max_info[j]:
                    max_info[j] = info_matrix[i,j]


@numba.jit(nopython=True, parallel=True)
def unbiased_pvalue_calc(info_matrix, unbiased_p_val, max_info):
    _populate_max_info_array(info_matrix, max_info)
    for i in numba.prange(info_matrix.shape[0]):
        # mi_actual = info_matrix[i,0]
        unbiased_p_val[i] = 0
        for j in numba.prange(max_info.shape[0]):
            if info_matrix[i,0] < max_info[j]:
                unbiased_p_val[i] += 1
        # Add 1 to both numerator and denominator
        # This account for the actual information value
        unbiased_p_val[i] += 1
        unbiased_p_val[i] = unbiased_p_val[i]/(max_info.shape[0]+1)
            

def unbiased_pvalue(info_matrix):
    max_info = np.zeros(info_matrix.shape[1] - 1, dtype=np.float32)
    unbiased_p_val = np.zeros(info_matrix.shape[0], dtype=np.float32)
    unbiased_pvalue_calc(info_matrix, unbiased_p_val, max_info)
    return unbiased_p_val


@numba.jit(nopython=True,parallel=True)
def p_median_calc(info_values, fold_tuples, median_values, p_median_values):
    # iterate through each row (independent variable)
    for i in numba.prange(info_values.shape[0]):
        # for each independent variable will need to compare the
        # in-sample information value (info_values) with the
        # median information_value in the corresponding tuple
        p_median_values[i] = 0
        for j in numba.prange(len(fold_tuples)):
            in_sample_idx = fold_tuples[j][0]
            oos_idx = fold_tuples[j][1]
            if info_values[i,in_sample_idx] < median_values[oos_idx]:
                p_median_values[i] += 1
        # actual p_median value is then the proportion of times the
        # in-sample actual information value is less than the
        # median out-of-sample information value across all
        # possible folds
        p_median_values[i] = p_median_values[i]/median_values.shape[0]