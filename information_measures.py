import numba
from numba import cuda
import math


@numba.jit(nopython=True)
def populate_marginals_grid(bins_x_array, bins_y_array, 
                            marginal_x_array, marginal_y_array,
                            grid_array):
    
    assert(len(bins_x_array) == len(bins_y_array))
    
    ncases = len(bins_x_array)
    
    nbins_x = len(marginal_x_array)
    nbins_y = len(marginal_y_array)

    # Populate the marginal array and grid matrix
    for i in range(ncases):
        ix = bins_x_array[i]
        iy = bins_y_array[i]
        marginal_x_array[ix] += 1
        marginal_y_array[iy] += 1
        grid_array[ix*nbins_y + iy] += 1


@numba.jit(nopython=True, parallel=True)
def ur_calc(ur_array, bins_x, bins_y_matrix, marginal_x, marginal_y, grid):

    for i in numba.prange(bins_y_matrix.shape[0]):
        
        populate_marginals_grid(bins_x, bins_y_matrix[i],
                                marginal_x[i], marginal_y[i], grid[i])

        nbins_x = marginal_x[i].shape[0]
        nbins_y = marginal_y[i].shape[0]

        # Test for single bin row or column
        if n_bins_x < 2 or nbins_y < 2:
            # Assign value of 0
            UR = 0.0
        else:                       
            # calculate number of cases
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[i,j]

            UR = ur_array[i]
            entropy_x = 0.0
            entropy_y = 0.0
            entropy_joint = 0.0
                    
            for j in range(nbins_x):
                px = marginal_x[i,j]/ncases
                entropy_x -= px * math.log(px)

            for k in range(nbins_y):
                py = marginal_y[i,k]/ncases
                entropy_y -= py * math.log(py)

            for j in range(nbins_x):
                for k in range(nbins_y):
                    pxy = grid[i,j*nbins_y + k]/ncases
                    entropy_joint -= pxy * math.log(pxy)
    
            if entropy_y > 0:
                UR = (entropy_x + entropy_y - entropy_joint) / entropy_y
            else:
                UR = 0.0
        
        ur_array[i] = UR

        
@numba.jit(nopython=True, parallel=True)
def ur_calc_parallel(ur_matrix, bins_x, bins_y, 
                     marginal_x, marginal_y, grid):
    
    for var in numba.prange(bins_x.shape[0]):
    
        for i in numba.prange(bins_y.shape[0]):

            populate_marginals_grid(bins_x[var], bins_y[i], 
                                    marginal_x[var,i], marginal_y[var,i],
                                    grid[var,i])

            nbins_x = marginal_x[var,i].shape[0]
            nbins_y = marginal_y[var,i].shape[0]

            # calculate number of cases first
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[var,i,j]

            # Test for single bin row or column
            if n_bins_x < 2 or nbins_y < 2:
                # Assign value of 0
                UR = 0.0
            else:    
                UR = ur_matrix[var,i]
                entropy_x = 0.0
                entropy_y = 0.0
                entropy_joint = 0.0

                for j in range(nbins_x):
                    px = marginal_x[var,i,j]/ncases
                    entropy_x -= px * math.log(px)
                    
                for k in range(nbins_y):
                    py = marginal_y[var,i,k]/ncases
                    entropy_y -= py * math.log(py)
                    
                for j in range(nbins_x):
                    for k in range(nbins_y):
                        pxy = grid[var,i,j*nbins_y + k]/ncases
                        entropy_joint -= pxy * math.log(pxy)

                if entropy_y > 0:
                    UR = (entropy_x + entropy_y - entropy_joint) / entropy_y
                else:
                    UR = 0.0
            
                ur_matrix[var,i] = UR


@numba.jit(nopython=True, parallel=True)
def mutinf_discrete_calc(mi_array, bins_x, bins_y_matrix, 
                         marginal_x, marginal_y, grid):
    
    for i in numba.prange(bins_y_matrix.shape[0]):
        
        populate_marginals_grid(bins_x, bins_y_matrix[i], 
                                marginal_x[i], marginal_y[i], grid[i])

        nbins_x = len(marginal_x[i])
        nbins_y = len(marginal_y[i])

        # calculate number of cases by summing either of the
        # marginal arrays
        ncases = 0
        for j in range(nbins_x):
            ncases += marginal_x[i,j]

        MI = mi_array[i]
            
        for j in range(nbins_x):
            px = marginal_x[i,j]/ncases
            for k in range(nbins_y):
                py = marginal_y[i,k]/ncases
                pxy = grid[i,j*nbins_y + k]/ncases
                if pxy > 0:
                    MI += pxy * math.log(pxy / (px*py))
        
        mi_array[i] = MI

    
@numba.jit(nopython=True, parallel=True)
def mutinf_discrete_calc_parallel(mi_matrix, bins_x, bins_y, 
                                  marginal_x, marginal_y, grid):
    
    for var in numba.prange(bins_x.shape[0]):
    
        for i in numba.prange(bins_y.shape[0]):

            populate_marginals_grid(bins_x[var], bins_y[i], 
                                    marginal_x[var,i], marginal_y[var,i],
                                    grid[var,i])

            nbins_x = marginal_x[var,i].shape[0]
            nbins_y = marginal_y[var,i].shape[0]

            # calculate number of cases first
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[var,i,j]

            MI = mi_matrix[var,i]

            for j in range(nbins_x):
                px = marginal_x[var,i,j]/ncases
                for k in range(nbins_y):
                    py = marginal_y[var,i,k]/ncases
                    pxy = grid[var,i,j*nbins_y + k]/ncases
                    if pxy > 0:
                        MI += pxy * math.log(pxy / (px*py))

            mi_matrix[var,i] = MI
    

@cuda.jit
def mutinf_discrete_calc_gpu(size, mi_array, bins_x, bins_y_matrix, 
                             marginal_x, marginal_y, grid):
    
    i = cuda.grid(1)
    
    if i < size:
    
        populate_marginals_grid(bins_x, bins_y_matrix[i],
                                marginal_x[i], marginal_y[i], grid[i])

        nbins_x = len(marginal_x[i])
        nbins_y = len(marginal_y[i])

        # calculate number of cases by summing either of the
        # marginal arrays
        ncases = 0
        for j in range(nbins_x):
            ncases += marginal_x[i,j]

        MI = mi_array[i]

        for j in range(nbins_x):
            px = marginal_x[i,j]/ncases
            for k in range(nbins_y):
                py = marginal_y[i,k]/ncases
                pxy = grid[i,j*nbins_y + k]/ncases
                if pxy > 0:
                    MI += pxy * math.log(pxy / (px*py))

        mi_array[i] = MI
        
