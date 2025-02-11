##############################################################################
# Fengjun Yang, 2022
# Utility functions for testing the learned value and tracking controllers
##############################################################################

import numpy as np

def relerr(A, Ahat):
    """ computes relative error of two matrices in terms of frobenius norm """
    return np.linalg.norm(A-Ahat, 'fro') / np.linalg.norm(A, 'fro')
