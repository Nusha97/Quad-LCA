from rotorpy.trajectories.traj_template import TrajTemplate, H_fun, get_1d_constraints, cvxopt_solve_qp
from scipy.linalg import block_diag
import numpy as np
import matplotlib.pyplot as plt
import cvxopt

class RegularizedTrajectory(TrajTemplate):
    def __init__(self, points, yaw_angles=None, yaw_rate_max=2*np.pi, 
                 poly_degree=7, yaw_poly_degree=7,
                 v_max=3, v_avg=1, v_start=[0, 0, 0], v_end=[0, 0, 0],
                 regularizer=None, verbose=True):
        """
        Waypoints and yaw angles compose the "keyframes" for optimizing over. 
        Inputs:
            points, numpy array of m 3D waypoints. 
            yaw_angles, numpy array of m yaw angles corresponding to each waypoint. 
            yaw_rate_max, the maximum yaw rate in rad/s
            v_avg, the average speed between waypoints, this is used to do the time allocation as well as impose constraints at midpoint of each segment. 
            v_start, the starting velocity vector given as an array [x_dot_start, y_dot_start, z_dot_start]
            v_end, the ending velocity vector given as an array [x_dot_end, y_dot_end, z_dot_end]
            use_neural_network, boolean, whether to use the neural network to get coefficients.
            regularizer, the regularizer function for the neural network.
            verbose, determines whether or not the QP solver will output information. 
        """
        if yaw_angles is None:
            self.yaw = np.zeros((points.shape[0]))
        else:
            self.yaw = yaw_angles
        self.v_avg = v_avg
        self.regularizer = regularizer

        cvxopt.solvers.options['show_progress'] = verbose

        # Compute the distances between each waypoint.
        seg_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        seg_mask = np.append(True, seg_dist > 1e-1)
        self.points = points[seg_mask, :]
        loc_points = self.points

        self.null = False

        m = loc_points.shape[0] - 1  # Get the number of segments

        # Compute the derivatives of the polynomials
        self.x_dot_poly = np.zeros((m, 3, poly_degree))
        self.x_ddot_poly = np.zeros((m, 3, poly_degree - 1))
        self.x_dddot_poly = np.zeros((m, 3, poly_degree - 2))
        self.x_ddddot_poly = np.zeros((m, 3, poly_degree - 3))
        self.yaw_dot_poly = np.zeros((m, 1, yaw_poly_degree))
        self.yaw_ddot_poly = np.zeros((m, 1, yaw_poly_degree - 1))

        # If two or more waypoints remain, solve min snap
        if loc_points.shape[0] >= 2:
            ################## Time allocation
            self.delta_t = seg_dist / self.v_avg  # Compute the segment durations based on the average velocity
            loc_delta_t = self.delta_t
            self.t_keyframes = np.concatenate(([0], np.cumsum(self.delta_t)))  # Construct time array which indicates when the quad should be at the i'th waypoint.

            ################## Cost function
            # First get the cost segment for each matrix:
            H_pos = [H_fun(loc_delta_t[i], k=poly_degree) for i in range(m)]
            H_yaw = [H_fun(loc_delta_t[i], k=yaw_poly_degree) for i in range(m)]

            # Now concatenate these costs using block diagonal form:
            P_pos = block_diag(*H_pos)
            P_yaw = block_diag(*H_yaw)

            # Lastly the linear term in the cost function is 0
            q_pos = np.zeros(((poly_degree + 1) * m, 1))
            q_yaw = np.zeros(((yaw_poly_degree + 1) * m, 1))

            ################## Constraints for each axis
            (Ax, bx, Gx, hx) = get_1d_constraints(
                loc_points[:, 0],
                loc_delta_t,
                m,
                k=poly_degree,
                vmax=v_max,
                vstart=v_start[0],
                vend=v_end[0],
            )
            (Ay, by, Gy, hy) = get_1d_constraints(
                loc_points[:, 1],
                loc_delta_t,
                m,
                k=poly_degree,
                vmax=v_max,
                vstart=v_start[1],
                vend=v_end[1],
            )
            (Az, bz, Gz, hz) = get_1d_constraints(
                loc_points[:, 2],
                loc_delta_t,
                m,
                k=poly_degree,
                vmax=v_max,
                vstart=v_start[2],
                vend=v_end[2],
            )
            (Ayaw, byaw, Gyaw, hyaw) = get_1d_constraints(
                self.yaw, loc_delta_t, m, k=yaw_poly_degree, vmax=yaw_rate_max
            )

            c_opt_x = cvxopt_solve_qp(P_pos, q=q_pos, G=Gx, h=hx, A=Ax, b=bx)
            c_opt_y = cvxopt_solve_qp(P_pos, q=q_pos, G=Gy, h=hy, A=Ay, b=by)
            c_opt_z = cvxopt_solve_qp(P_pos, q=q_pos, G=Gz, h=hz, A=Az, b=bz)
            c_opt_yaw = cvxopt_solve_qp(P_yaw, q=q_yaw, G=Gyaw, h=hyaw, A=Ayaw, b=byaw)

            # call modify_reference directly after computing the min snap coeffs and use the returned coeffs in the rest of the class
            self.nan_encountered = False

            min_snap_coeffs = np.concatenate([c_opt_x, c_opt_y, c_opt_z, c_opt_yaw])
            # get H by concatenating H_pos and H_yaw
            H = block_diag(
                *[
                    0.5 * (P_pos.T + P_pos),
                    0.5 * (P_pos.T + P_pos),
                    0.5 * (P_pos.T + P_pos),
                    0.5 * (P_yaw.T + P_yaw),
                ]
            )  # cost function is the same for x, y, z

            # get A by concatenating Ax, Ay, Az, Ayaw
            A = block_diag(*[Ax, Ay, Az, Ayaw])

            # get b by concatenating bx, by, bz, byaw
            b = np.concatenate((bx, by, bz, byaw))

            self.H = H
            self.A = A
            self.b = b
            self.min_snap_coeffs = min_snap_coeffs

            nn_coeff, pred, nan_encountered = sgd_jax.modify_reference(
                regularizer,
                H,
                A,
                b,
                min_snap_coeffs
            )

            self.nan_encountered = nan_encountered  # Update the value

            if not nan_encountered:
                c_opt_x = nn_coeff[0:((poly_degree + 1) * m)]
                c_opt_y = nn_coeff[((poly_degree + 1) * m):(2 * (poly_degree + 1) * m)]
                c_opt_z = nn_coeff[(2 * (poly_degree + 1) * m):(3 * (poly_degree + 1) * m)]
                c_opt_yaw = nn_coeff[(3 * (poly_degree + 1) * m):]

            self.c_opt_x = c_opt_x
            self.c_opt_y = c_opt_y
            self.c_opt_z = c_opt_z
            self.c_opt_xyz = np.concatenate([c_opt_x, c_opt_y, c_opt_z])
            self.c_opt_yaw = c_opt_yaw

            ################## Construct polynomials from c_opt
            self.x_poly = np.zeros((m, 3, (poly_degree + 1)))
            self.yaw_poly = np.zeros((m, 1, (yaw_poly_degree + 1)))
            for i in range(m):
                self.x_poly[i, 0, :] = np.flip(c_opt_x[(poly_degree + 1) * i:((poly_degree + 1) * i + (poly_degree + 1))])
                self.x_poly[i, 1, :] = np.flip(c_opt_y[(poly_degree + 1) * i:((poly_degree + 1) * i + (poly_degree + 1))])
                self.x_poly[i, 2, :] = np.flip(c_opt_z[(poly_degree + 1) * i:((poly_degree + 1) * i + (poly_degree + 1))])
                self.yaw_poly[i, 0, :] = np.flip(c_opt_yaw[(yaw_poly_degree + 1) * i:((yaw_poly_degree + 1) * i + (yaw_poly_degree + 1))])

            for i in range(m):
                for j in range(3):
                    self.x_dot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=1)
                    self.x_ddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=2)
                    self.x_dddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=3)
                    self.x_ddddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=4)
                self.yaw_dot_poly[i, 0, :] = np.polyder(self.yaw_poly[i, 0, :], m=1)
                self.yaw_ddot_poly[i, 0, :] = np.polyder(self.yaw_poly[i, 0, :], m=2)

        else:
            # Otherwise, there is only one waypoint so we just set everything = 0.
            self.null = True
            m = 1
            self.T = np.zeros((m,))
            self.x_poly = np.zeros((m, 3, 6))
            self.x_poly[0, :, -1] = points[0, :]

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.
        Inputs:
            t, time, s
        Outputs:
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        yaw_ddot = 0

        if self.null:
            x = self.points[0, :]
            yaw = self.yaw[0]
        else:
            t = np.clip(t, self.t_keyframes[0], self.t_keyframes[-1])
            for i in range(self.t_keyframes.size - 1):
                if self.t_keyframes[i] + self.delta_t[i] >= t:
                    break
            t = t - self.t_keyframes[i]

            for j in range(3):
                x[j] = np.polyval(self.x_poly[i, j, :], t)
                x_dot[j] = np.polyval(self.x_dot_poly[i, j, :], t)
                x_ddot[j] = np.polyval(self.x_ddot_poly[i, j, :], t)
                x_dddot[j] = np.polyval(self.x_dddot_poly[i, j, :], t)
                x_ddddot[j] = np.polyval(self.x_ddddot_poly[i, j, :], t)

            yaw = np.polyval(self.yaw_poly[i, 0, :], t)
            yaw_dot = np.polyval(self.yaw_dot_poly[i, 0, :], t)
            yaw_ddot = np.polyval(self.yaw_ddot_poly[i, 0, :], t)

        flat_output = {
            'x': x,
            'x_dot': x_dot,
            'x_ddot': x_ddot,
            'x_dddot': x_dddot,
            'x_ddddot': x_ddddot,
            'yaw': yaw,
            'yaw_dot': yaw_dot,
            'yaw_ddot': yaw_ddot
        }
        return flat_output
    
    def evaluate_trajectory(self, times):
        """
        Evaluates the minsnap trajectory throughout a time interval given by times.
        Input:
            times, an array (N,) of N time points. We assume that times is in ascending order.
        Output:
            flat_outputs: the outputs of the trajectory in the order (pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot)
        """
        N = times.shape[0]

        flat_outputs = []

        for t in times:
            flat = self.update(t)

            x = flat["x"]
            x_dot = flat["x_dot"]
            x_ddot = flat["x_ddot"]
            x_dddot = flat["x_dddot"]
            x_ddddot = flat["x_ddddot"]
            yaw = np.array(flat["yaw"])
            yaw_dot = np.array(flat["yaw_dot"])
            yaw_ddot = np.array(flat["yaw_ddot"])

            flat_outputs.append(
                np.concatenate(
                    (
                        x,
                        x_dot,
                        x_ddot,
                        x_dddot,
                        x_ddddot,
                        np.array([yaw, yaw_dot, yaw_ddot]),
                    )
                )
            )

        return np.array(flat_outputs)
    
