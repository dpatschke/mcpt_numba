import numpy as np

import numba

from numba import cuda


@numba.jit(nopython=True, nogil=True)
def reorder_jit(arr, idx):

    n = len(arr)
    
    for current_idx in range(n):
        swap_idx = idx[current_idx]
        while swap_idx < current_idx:
            swap_idx = idx[swap_idx]
        arr[current_idx], arr[swap_idx] = arr[swap_idx], arr[current_idx]


@numba.jit(nopython=True, nogil=True)
def rank_jit(arr, rank_idx):
    
    n = len(arr)
    
    rank_idx[0] = 0
    k = 0
    
    for i in range(n):
        i += 1
        if (arr[i] - arr[i-1]) >= (1e-12 * (1 + abs(arr[i]) + abs(arr[i-1]))):
            k += 1
        rank_idx[i] = k


# The following argsort code was taken from the following gist:
# https://gist.github.com/hgrecco/818705ad1eef70f5f6ab
@numba.jit(nopython=True, nogil=True)
def partition_gpu(values, idxs, left, right):
    """
    Partition method
    """
    piv = values[idxs[left]]
    i = left + 1
    j = right

    while True:
        while i <= j and values[idxs[i]] <= piv:
            i += 1
        while j >= i and values[idxs[j]] >= piv:
            j -= 1
        if j <= i:
            break

        idxs[i], idxs[j] = idxs[j], idxs[i]

    idxs[left], idxs[j] = idxs[j], idxs[left]
    
    return j


@numba.jit(nopython=True, nogil=True)
def qsortdsi_gpu(values, idxs, tmp):

    left = 0
    right = len(values) - 1

    ndx = 0

    tmp[ndx, 0] = left
    tmp[ndx, 1] = right

    ndx = 1
    while ndx > 0:

        ndx -= 1
        right = tmp[ndx, 1]
        left = tmp[ndx, 0]

        piv = partition_gpu(values, idxs, left, right)

        if piv - 1 > left:
            tmp[ndx, 0] = left
            tmp[ndx, 1] = piv - 1
            ndx += 1

        if piv + 1 < right:
            tmp[ndx, 0] = piv + 1
            tmp[ndx, 1] = right
            ndx += 1
    
    reorder_jit(values, idxs)


@numba.jit(nopython=True, nogil=True)
def qsortdsi_cpu(values, idxs):
    sorted_idxs = np.argsort(values)
    reorder_jit(values, sorted_idxs)
    # copy over values from sorted_idxs to idxs
    for i in range(idxs.shape[0]):
        idxs[i] = sorted_idxs[i]


@numba.jit(nopython=True, nogil=True)
def compute_initial_bounds(bin_end, n_values):
    
    n_bins = bin_end.shape[0]
    k = 0 

    # For all partitions, find the number of cases in the partition
    # advance the index of next one up and store upper bound in the
    # bin_end array
    for i in range(n_bins):
        j = int((n_values - k) / (n_bins - i))
        k += j
        bin_end[i] = k - 1


@numba.jit(nopython=True, nogil=True)
def adjust_bounds(bin_end, rank_idx, n_bins_array):
    """
    If the data has no ties, the partitioning is complete.
    However, if there are ties, we must iterate until no 
    partition boundary splits a tie.
    
    Note that the upper bound of the last partition is
    always the last case in the sorted array, so we don't
    need to worry about it splitting a tie. There are no cases
    above it!  All we care about are the np-1 internal 
    boundaries.
    """
    n_bins = n_bins_array[0]
    
    while True:

        tie_found = 0

        for ibound in range(n_bins-1):
            if rank_idx[bin_end[ibound]] == rank_idx[bin_end[ibound]+1]:
                # This bound splits a tie.  Remove this bound.
                for i in range(ibound+1, n_bins):
                    bin_end[i-1] = bin_end[i]
                # Adjust the number of bins down by 1
                n_bins -= 1
                # Flag that a tie was found
                tie_found = 1
                break

        # Break out of the loop if no tie is found
        if tie_found == 0:
            break
        
        # The offending bound is now gone.
        # Try splitting each remaining bin. 
        # For each split, check the size of the smaller resulting bin.
        # Choose the split that gives the largest of the smaller.
        # Note that n_bins has been decremented, so now n_bins is less
        # than what was originally desired
        istart = 0
        nbest = -1
        
        for ibound in range(n_bins):
            istop = bin_end[ibound]
            # Now processing a bin from istart through istop, inclusive
            # Try all possible splits of the bin.
            # If it splits a tie, don't check
            for i in range(istart, istop):
                if rank_idx[i] == rank_idx[i+1]:
                    continue
                nleft = i - istart + 1      # Number of cases in left half
                nright = istop - i          # Number of cases in right half
                if nleft < nright:
                    if nleft > nbest:
                        nbest = nleft
                        ibound_best = ibound
                        isplit_best = i
                else:
                    if nright > nbest:
                        nbest = nright
                        ibound_best = ibound
                        isplit_best = i

            istart = istop + 1
        
        # The search is done.  It may (rarely) be the case that no further
        # splits are possible.  This will happen if the user requests more
        # partitions than there are unique values in the dataset. We know 
        # that this has happened if nbest is still -1.  In this case, we
        # (obviously) cannot do a split to make up for the one lost above.
        if (nbest < 0):
            continue
    
        # We get here when the best split of an existing partition has been
        # found.  Save it.  The bin that we are splitting is ibound_best,
        # and the split for a new bound is at isplit_best.
        ibound = n_bins - 1
        while ibound >= ibound_best:
            bin_end[ibound+1] = bin_end[ibound]
            ibound -= 1
        
        bin_end[ibound_best] = isplit_best
        n_bins += 1
        
    n_bins_array[0] = n_bins


@numba.jit(nopython=True, nogil=True)
def determine_bin_bound_values(bounds, x, bin_end, n_bins):
    
    for ibound in range(n_bins):
        bounds[ibound] = x[bin_end[ibound]]


@numba.jit(nopython=True, nogil=True)
def assign_bin_membership(bins, idxs, bin_end, n_bins):

    istart = 0
    
    for ibound in range(n_bins):
        istop = bin_end[ibound]
        for i in range(istart, istop+1):
            bins[idxs[i]] = ibound
        istart = istop + 1


def set_binning_variables(data, nbins, target='cpu'):
    # If array is a single dimension, then convert it
    # to a two-dimensional matrix, so that parallel
    # operations can be performed on each row
    if data.ndim == 1:
        ncols = 1
        nobs = data.shape[0]
        data_copy = np.reshape(data.copy(), (1, nobs))
    else:
        assert data.ndim == 2, \
            "Functions only valid on 1d or 2d data."
        # data needs to be transposed so that each
        # row represents a column in the data set
        # Necessary for parallel operations on the GPU
        data_copy = np.transpose(data).copy()
        ncols = data_copy.shape[0]
        nobs = data_copy.shape[1]
    
    idxs = np.arange(nobs)
    idxs = np.tile(idxs, (ncols,1))
    
    max_depth = np.int32((nobs - 1) / 2)
    tmp = np.zeros((ncols, max_depth, 2), dtype=np.int32)
    
    n_bins = np.repeat(nbins, ncols).astype(np.int16)
    n_bins = np.reshape(n_bins, (ncols,1))
    
    bins = np.zeros((ncols, nobs), dtype=np.int32)
    bounds = np.zeros((ncols, nbins), dtype=np.int32)
    bin_end = np.zeros((ncols, nbins), dtype=np.int32)
    rank_idx = np.zeros((ncols, nobs), dtype=np.int32)
    
    # Create a dictionary that is going to store all the
    # values needed to perform the binning. If the target
    # is the gpu then transfer the objects to the gpu and
    # assign the cuda objects to the dictionary.
    bin_vars_dict = {}
    
    if target == 'gpu':
        # Push the arrays to the GPU device and assign their 
        # variables to variables in the bin_vars_dict
        bin_vars_dict['data_copy'] = cuda.to_device(data_copy)
        bin_vars_dict['idxs'] = cuda.to_device(idxs)
        bin_vars_dict['tmp'] = cuda.to_device(tmp)
        bin_vars_dict['n_bins'] = cuda.to_device(n_bins)
        bin_vars_dict['bins'] = cuda.to_device(bins)
        bin_vars_dict['bounds'] = cuda.to_device(bounds)
        bin_vars_dict['bin_end'] = cuda.to_device(bin_end)
        bin_vars_dict['rank_idx'] = cuda.to_device(rank_idx)
    else:
        # default to CPU
        bin_vars_dict['data_copy'] = data_copy
        bin_vars_dict['idxs'] = idxs
        bin_vars_dict['tmp'] = tmp
        bin_vars_dict['n_bins'] = n_bins
        bin_vars_dict['bins'] = bins
        bin_vars_dict['bounds'] = bounds
        bin_vars_dict['bin_end'] = bin_end
        bin_vars_dict['rank_idx'] = rank_idx
    
    return bin_vars_dict


@numba.jit(nopython=True, parallel=True)
def masters_cut_cpu(data_copy, n_bins_array, bounds, bins,
                    rank_idx, bin_end, idxs):
    
    for i in numba.prange(data_copy.shape[0]):
        n = data_copy[i].shape[0]
        
        qsortdsi_cpu(data_copy[i], idxs[i])
        rank_jit(data_copy[i], rank_idx[i])
        compute_initial_bounds(bin_end[i], n)
        adjust_bounds(bin_end[i], rank_idx[i], n_bins_array[i])
        
        n_bins = n_bins_array[i,0]
        
        determine_bin_bound_values(bounds[i], data_copy[i], bin_end[i], n_bins)
        assign_bin_membership(bins[i], idxs[i], bin_end[i], n_bins)


@cuda.jit
def masters_cut_gpu(data_copy, n_bins_array, bounds, bins,
                    rank_idx, bin_end, idxs, tmp):

    n = len(data_copy)
    qsortdsi_gpu(data_copy, idxs, tmp)
    rank_jit(data_copy, rank_idx)
    compute_initial_bounds(bin_end, n)
    adjust_bounds(bin_end, rank_idx, n_bins_array)
    n_bins = n_bins_array[0]
    determine_bin_bound_values(bounds, data_copy, bin_end, n_bins)
    assign_bin_membership(bins, idxs, bin_end, n_bins)


def masters_cut(array, nbins=5, target='cpu'):
    # create the dictionary with all the variables necessary
    # to perform the masters cut partitioning
    bin_var_dict = set_binning_variables(array, nbins=nbins, target=target)
    
    # determine whether to run the cpu or gpu version
    # based on the elements in bin_var_dict
    if isinstance(bin_var_dict['data_copy'], np.ndarray):
        masters_cut_cpu(data_copy = bin_var_dict['data_copy'],
                       n_bins_array = bin_var_dict['n_bins'],
                       bounds = bin_var_dict['bounds'],
                       bins = bin_var_dict['bins'],
                       rank_idx = bin_var_dict['rank_idx'],
                       bin_end = bin_var_dict['bin_end'],
                       idxs = bin_var_dict['idxs'])
    else:
        masters_cut_gpu(bin_var_dict['data_copy'],
                        bin_var_dict['n_bins'],
                        bin_var_dict['bounds'],
                        bin_var_dict['bins'],
                        bin_var_dict['rank_idx'],
                        bin_var_dict['bin_end'],
                        bin_var_dict['idxs'],
                        bin_var_dict['tmp'])

    # Delete all the  objects in bin_var_dict that are no longer needed
    # after masters_cut is complete in order to free up memory
    del bin_var_dict['data_copy']
    del bin_var_dict['bounds']
    del bin_var_dict['rank_idx']
    del bin_var_dict['bin_end']
    del bin_var_dict['idxs']
    del bin_var_dict['tmp']

    return bin_var_dict
