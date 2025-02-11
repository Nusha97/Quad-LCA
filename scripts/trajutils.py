import jax.numpy as jnp
import numpy as np
#import scipy.special as sps
import jax.scipy.special as sps
# import numpy as np

def _diff_coeff(n, t, dx_order):
    ''' Returns a vector v whose inner product with the coefficients
            <v, c> = p^(dx_order)(t), where c = [c_n, ..., c_1]
    Input:
        - n:        Integer, order of the polynomial
        - t:        Float, point at which the polynomial is evaluated
        - dx_order: Integer, order of the differentiation
    Return:
        - v:        the vector satisfying the above equation
    '''
    assert dx_order <= n
    count = n - dx_order + 1 # Number of nonzero elements
    rrange = jnp.arange(count-1, -1, -1)  # Reversed range
    t_powers = t ** rrange
    multln = sps.gammaln(rrange + dx_order + 1) - sps.gammaln(rrange+1)
    mult = jnp.exp(multln)
    v_reverse = jnp.concatenate([t_powers * mult, jnp.zeros(dx_order)])
    return v_reverse

def _facln(n, i):
    ''' Helper function that computes ln( n*(n-i)*...*(n-i+1) ) '''
    return jnp.log(n - jnp.arange(i)).sum()

def _cost_matrix(n, k, T):
    ''' Return the quadratic cost matrix corresponds to the cost function
            \int_0^T (\frac{\partial^k x}{\partial t^k})^2 dt
    Input:
        - n: degree of the polynomial
        - k: order of derivative that we integrate over
    Return:
        - H: cost matrix
    '''
    H = jnp.zeros((n+1, n+1))
    for i in range(n-k+1):
        for j in range(n-k+1):
            power = 2*n-2*k-i-j+1
            Hij_ln = _facln(n-i, k) + _facln(n-j, k) - jnp.log(power)
            #print(H[i, j].shape)
            #print(Hij_ln.shape)
            #print(T.shape)
            #H[i, j] = jnp.exp(Hij_ln + jnp.log(T) * power)
            H = H.at[i, j].set(np.exp(Hij_ln + np.log(T) * power))
    return jnp.array(H)

