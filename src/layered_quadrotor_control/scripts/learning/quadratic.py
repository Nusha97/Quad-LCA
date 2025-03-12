import numpy as np
import cvxpy as cp

from .trajutils import _diff_coeff, _facln, _cost_matrix

def _continuity_constr(n, order, coeff1, coeff2, x_wp, T1):
    ''' Return a list of continuity constraints enforced at p_1(T1) and p_2(0).
    In addition, enforce waypoint bc at the two points.
    Input:
        - n:        Integer, order of the polynomial
        - order:    Integer, order of continuity enforced
        - coeff1:   cp.Variable, coefficients of polynomial p_1
        - coeff2:   cp.Variable, coefficients of polynomial p_2
        - x_wp:     Float, waypoint
        - T1:       Float, endpoint of polynomial p_1
    Return:
        - constr:   list of cp.Constraint
    '''
    # Waypoint constraint
    if (x_wp is not None) or (not np.isnan(x_wp)):
        wp_constr = [
            _diff_coeff(n, T1, 0) @ coeff1 == x_wp,
            _diff_coeff(n, 0, 0)  @ coeff2 == x_wp
        ]
    else:
        wp_constr = [
            _diff_coeff(n, T1, 0) @ coeff1 == _diff_coeff(n, 0, 0)  @ coeff2
        ]
    # Continuity constraint
    cont_constr = [
            _diff_coeff(n, T1, i) @ coeff1 == _diff_coeff(n, 0, i) @ coeff2 \
            for i in range(1, order)
    ]
    return wp_constr + cont_constr


def _boundary_cond(n, coeff, bcs, T):
    ''' Return a list of bc constraints enforced at p(T)
    Input:
        - n:        Integer, order of polynomial
        - coeff:    coefficients of the polynomial p(t)
        - bcs:      list of boundary condition values, ordered as
                    [p(0), p'(0), p''(0), ...]
    Return:
        - constr:   list of cp.Constraint
    '''
    constr = []
    for i, bc in enumerate(bcs):
        if (bc is not None) or (not np.isnan(bc)):
            constr.append( _diff_coeff(n, T, i) @ coeff == bc )
    return constr

def min_jerk_1d(waypoints, ts, n, num_steps):
    ''' Generate the min-jerk trajectory
    Input:
        - waypoints:    list of 1d waypoints
        - ts:           list of times for the waypoints (ascending order)
        - n:            order of the polynomial
        - num_steps:    number of timesteps used to sample the traj
    Return:
        - obj:          cp.Expression, the min-jerk objective
        - constr:       list of cp.Constraint, the min-jerk constraints
        - ref:          cp.Variable, the reference trajectory
        - coeff:        cp.Variable, the coefficients of the polynomials
    '''
    # Define variables
    coeff = cp.Variable((len(waypoints)-1, n+1))
    ref = cp.Variable(num_steps)
    objective = 0
    constr = []
    # Duration for each segment
    ts = np.array(ts)
    durations = ts[1:] - ts[:-1]
    # Compute objective
    for i in range(len(waypoints)-1):
        H = _cost_matrix(n, 3, durations[i])
        objective += cp.quad_form(coeff[i], H)
    # Boundary conditions
    constr += _boundary_cond(n, coeff[0], [waypoints[0], 0, 0], 0)
    constr += _boundary_cond(n, coeff[-1], [waypoints[-1], 0, 0], durations[-1])
    # Continuity constraints
    for i in range(len(waypoints)-2):
        constr += _continuity_constr(n, 3, coeff[i], coeff[i+1],
                                     waypoints[i+1], durations[i])
    # Construct reference from coeff
    k = 0
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], num_steps)):
        if tt > ts[k+1]: k += 1
        constr += [
                ref[i] == _diff_coeff(n, tt-ts[k], 0) @ coeff[k]
        ]
    return objective, constr, ref, coeff

def min_jerk_setup(waypoints, ts, n, p, num_steps):
    ''' Sets up the min jerk problem by calling the 1d helper function '''
    objective, constrs, refs, coeffs = 0, [], [], []
    # Generate the variables and constraints for each dimension
    for i in range(p):
        o, c, r, co = min_jerk_1d([w[i] for w in waypoints], ts, n, num_steps)
        objective += o
        constrs += c
        refs.append(r)
        coeffs.append(co)
    # Stitch them into global reference trajectory
    ref = cp.vstack(refs).flatten()
    return objective, constrs, ref, coeffs


def generate(waypoints, ts, n, num_steps, p, P, rho, task='min-jerk'):
    ''' Wrapper for generating trajectory. For now only takes quadratic
    regularization.
    Return:
        ref:        reference trajectory
    '''
    #import ipdb;
    #ipdb.set_trace()
    objective, constr, ref, coeff = min_jerk_setup(waypoints, ts, n, p, num_steps)
    if P is None:
        penalty = 0
    else:
        P12 = P[0:p,p:]
        P22 = P[p:,p:]
        x0 = np.array(waypoints[0])
        penalty = cp.quad_form(ref, P22) + 2 * x0 @ (P12@ref)
    prob = cp.Problem(cp.Minimize(objective + rho * penalty), constr)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        return None
    else:
        return ref.value, np.array([c.value for c in coeff])


# Changes made by Anusha S.
def traj_coeffs(waypoints, ts, n, num_steps, p, P, rho, task='min-jerk'):
    """ Generating trajectories and passing the coefficients to compute the
    full state of the quad-rotor system
    Return:
        coeff:        polynomial coeffcients
    """
    objective, constr, ref, coeff = min_jerk_setup(waypoints, ts, n, p, num_steps)
    P12 = P[0:p, p:]
    P22 = P[p:, p:]

    x0 = np.array(waypoints[0])
    penalty = cp.quad_form(ref, P22) + 2 * x0 @ (P12 @ ref)
    prob = cp.Problem(cp.Minimize(objective + rho * penalty), constr)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        return None
    else:
        return coeff
