"""
To perform inference on an experiment that collected 20 trajectories and compile all data
"""

from construct_traj_list import *
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
import torch


def load_infbags(str1, str2, str3, rho):
    # Load inference bags with Lissajous curves
    bag = load_bag(str1)
    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(bag, str2, str3, rho)
    bag.close()
    return ref_traj, actual_traj, input_traj, cost_traj, times


def compute_actual_cost(start, end, ref_traj, actual_traj, input_traj):
    total_cost = []

    for i in range(len(start)):
        #total_cost = np.append(total_cost, sum(cost[start[i]:end[i]]))
        print(actual_traj[range(start[i], end[i])])
        total_cost.append(compute_tracking_cost(ref_traj[:, i*4:(i+1)*4],np.array(actual_traj[range(start[i], end[i])]), np.array(input_traj[range(start[i], end[i])])))
    return np.sum(total_cost)


def compute_tracking_cost(ref_traj, actual_traj, input_traj):
    #xcost = np.linalg.norm(actual_traj[:, (0, 1, 2)] - ref_traj[:, (0, 1, 2)], axis=1) ** 2
    if actual_traj.shape[0] < ref_traj.shape[0]:
        actual_traj = np.append(actual_traj, np.zeros((4, ref_traj.shape[0]-actual_traj.shape[0]))).reshape(ref_traj.shape)
    xcost = np.linalg.norm(actual_traj[:, 0:3] - ref_traj[:, 0:3], axis=1) ** 2
    ucost = 0.1 * np.linalg.norm(input_traj, axis=1) ** 2
    # Cost from yaw (needs special treatment because quotient norm)
    ar = np.abs(actual_traj[:, 3] - ref_traj[:, 3])
    ra = np.abs(actual_traj[:, 3] + 2 * np.pi - ref_traj[:, 3])
    yawcost = np.minimum(ar, ra) ** 2
    # cost = xcost + yawcost + ucost
    return xcost + yawcost


def compute_total_cost(start, end, cost):
    total_cost = []
    for i in range(len(start)):
        #total_cost = np.append(total_cost, sum(cost[start[i]:end[i]]))
        total_cost.append(np.sum(cost[start[i]:end[i]]))
    return total_cost



def load_object(str):
    """
    Function to load to a pickle file
    :param str:
    :return:
    """
    with open(str, 'rb') as handle:
        return pickle.load(handle)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def generate_polynomial_trajectory(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = np.linspace(0, 1, T)

    # Solve for the polynomial coefficients
    coeffs = np.zeros(order + 1)
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    polynomial = np.zeros(T)
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return trajectory


def generate_polynomial_coeffs(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = np.linspace(0, 1, T)

    # Solve for the polynomial coefficients
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return coeffs


def compute_coeff_deriv(coeff, n, segments):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    coeff_new = coeff.copy()
    for i in range(segments):  # piecewise polynomial
        for j in range(n):  # Compute nth derivative of polynomial
            t = np.poly1d(coeff_new[i, :]).deriv()
            coeff_new[i, j] = 0
            coeff_new[i, j+1:] = t.coefficients
    return coeff_new


def sampler(poly, T, ts):
    """
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    k = 0
    ref = []
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], T)):
        if tt > ts[k + 1]: k += 1
        ref.append(poly[k](tt-ts[k]))
    return ref


def compute_pos_vel_acc(Tref, nn_coeffs, segments, ts):
    """
    Function to compute pos, vel, acc from nn coeffs
    :param timesteps:
    :return:
    """
    # Compute full state
    coeff_x = np.vstack(nn_coeffs[0, :, :])
    coeff_y = np.vstack(nn_coeffs[1, :, :])
    coeff_z = np.vstack(nn_coeffs[2, :, :])
    coeff_yaw = np.vstack(nn_coeffs[3, :, :])

    pos = []
    vel = []
    acc = []
    jerk = []

    x_ref = [np.poly1d(coeff_x[i, :]) for i in range(segments)]
    x_ref = np.vstack(sampler(x_ref, Tref, ts)).flatten()

    y_ref = [np.poly1d(coeff_y[i, :]) for i in range(segments)]
    y_ref = np.vstack(sampler(y_ref, Tref, ts)).flatten()

    z_ref = [np.poly1d(coeff_z[i, :]) for i in range(segments)]
    z_ref = np.vstack(sampler(z_ref, Tref, ts)).flatten()
    pos.append([x_ref, y_ref, z_ref])

    dot_x = compute_coeff_deriv(coeff_x, 1, segments)
    xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(segments)]
    xdot_ref = np.vstack(sampler(xdot_ref, Tref, ts)).flatten()

    dot_y = compute_coeff_deriv(coeff_y, 1, segments)
    ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(segments)]
    ydot_ref = np.vstack(sampler(ydot_ref, Tref, ts)).flatten()

    dot_z = compute_coeff_deriv(coeff_z, 1, segments)
    zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(segments)]
    zdot_ref = np.vstack(sampler(zdot_ref, Tref, ts)).flatten()
    vel.append([xdot_ref, ydot_ref, zdot_ref])

    ddot_x = compute_coeff_deriv(coeff_x, 2, segments)
    xddot_ref = [np.poly1d(ddot_x[i, :]) for i in range(segments)]
    xddot_ref = np.vstack(sampler(xddot_ref, Tref, ts)).flatten()

    ddot_y = compute_coeff_deriv(coeff_y, 2, segments)
    yddot_ref = [np.poly1d(ddot_y[i, :]) for i in range(segments)]
    yddot_ref = np.vstack(sampler(yddot_ref, Tref, ts)).flatten()

    ddot_z = compute_coeff_deriv(coeff_z, 2, segments)
    zddot_ref = [np.poly1d(ddot_z[i, :]) for i in range(segments)]
    zddot_ref = np.vstack(sampler(zddot_ref, Tref, ts)).flatten()
    acc.append([xddot_ref, yddot_ref, zddot_ref])

    dddot_x = compute_coeff_deriv(coeff_x, 3, segments)
    xdddot_ref = [np.poly1d(dddot_x[i, :]) for i in range(segments)]
    xdddot_ref = np.vstack(sampler(xdddot_ref, Tref, ts)).flatten()

    dddot_y = compute_coeff_deriv(coeff_y, 3, segments)
    ydddot_ref = [np.poly1d(dddot_y[i, :]) for i in range(segments)]
    ydddot_ref = np.vstack(sampler(ydddot_ref, Tref, ts)).flatten()

    dddot_z = compute_coeff_deriv(coeff_z, 3, segments)
    zdddot_ref = [np.poly1d(dddot_z[i, :]) for i in range(segments)]
    zdddot_ref = np.vstack(sampler(zdddot_ref, Tref, ts)).flatten()
    jerk.append([xdddot_ref, ydddot_ref, zdddot_ref])

    yaw_ref = [np.poly1d(coeff_yaw[i, :]) for i in range(segments)]
    yaw_ref = np.vstack(sampler(yaw_ref, Tref, ts)).flatten()

    dot_yaw = compute_coeff_deriv(coeff_yaw, 1, segments)
    yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
    yawdot_ref = np.vstack(sampler(yawdot_ref, Tref, ts)).flatten()

    #print("Pos array shape", np.array(pos).shape)

    return np.vstack(pos), np.vstack(vel), np.vstack(acc), np.vstack(jerk), yaw_ref, yawdot_ref


def generate_lissajous_traj(s, x_num_periods, y_num_periods, z_num_periods, yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp):
    """
    Function to generate Lissajous trajectory
    :return:
    """
    PI = np.pi
    x = lambda a: x_amp * (1 - np.cos(2 * PI * x_num_periods * a / period))
    y = lambda a: y_amp * np.sin(2 * PI * y_num_periods * a / period)
    z = lambda a: z_amp * np.sin(2 * PI * z_num_periods * a / period)
    yaw = lambda a: yaw_amp * np.sin(2 * PI * yaw_num_periods * a / period)
    return [x(s), y(s), z(s), yaw(s)]


def main():
    # Write a loop to iterate through different bags and compute costs and make plots

    np.random.seed(89)

    num_traj = 10
    dim = 4
    horizon = 300

    ref_coeff = np.zeros([num_traj, dim, horizon])

    rhos = [0, 1, 5, 10, 20, 50, 100]

    for i in range(num_traj):
        # Testing on Lissajous curve

        x_amp = 0.65 * np.random.random()
        y_amp = 0.55 * np.random.random()
        z_amp = 0.55 * np.random.random()
        yaw_amp = 0.6414 * np.random.random()
        x_num_periods = 1
        y_num_periods = 1
        z_num_periods = 1
        yaw_num_periods = 1
        period = 3

        ref = generate_lissajous_traj(np.linspace(0, 3, 301), x_num_periods, y_num_periods, z_num_periods,
                                      yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp)


        segments = 6
        ts = np.linspace(0, 3, segments + 1)
        order = 4
        # Tref = 300
        waypt = np.array(ref)[:, 0::50]
        waypt[:, 3] = waypt[:, 3] + 0.3
        offset = min(waypt[2, :])
        waypt[2, :] = waypt[2, :] - offset + 1

        # ref_coeff, init_coeff = quadratic.generate(waypt.T, ts, order, horizon, dim, None, 0)

        for k in range(dim):
            for j in range(segments):
                ref_coeff[i, k, j*50:(j+1)*50] = generate_polynomial_trajectory(waypt[k, j], waypt[k, j + 1],
                                                                                         50, order)

        """init_coeff = np.zeros([dim, segments, order + 1])

        for k in range(dim):
            for j in range(segments):
                init_coeff[k, j, :] = generate_polynomial_coeffs(waypt[k, j], waypt[k, j + 1], 50, order)

        ref_coeff = np.zeros((horizon, dim))
        pos, _, _, _, yaw, _ = compute_pos_vel_acc(horizon, init_coeff, segments, ts)
        ref_coeff[:, 0:3] = pos.T
        ref_coeff[:, 3] = yaw"""

    traj_costs = []

    for i in range(3):
        if i == 0:
            #ref_traj, actual_traj, input_traj, cost_traj, times_init = load_infbags('/home/anusha/poly_bag2.bag', 'dragonfly1',
            #                                                                        "/home/anusha/poly_ref_times.pkl", 1)
            #ref_traj = load_object("/home/anusha/poly_ref.pkl")
            #ref_traj = np.vstack(ref_coeff).T
            #ref_traj = ref_coeff

            #ref_traj, actual_traj, input_traj, cost_traj, times_init = load_infbags('/home/anusha/dragonfly1-poly.bag', 'dragonfly1',
            #                                                                        "/home/anusha/poly_inf_times.pkl", 0)
            ref_traj, actual_traj, input_traj, cost_traj, times_init = load_infbags('/home/anusha/replan/dragonfly1-mj.bag',
                                                                                    'dragonfly1',
                                                                                    "/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_mj.pkl",
                                                                                    0)
            print("Ref traj", ref_traj.shape)
            print(len(actual_traj))
            total_cost = np.linalg.norm(np.vstack(ref_traj) - np.vstack(actual_traj))
            traj_costs.append(total_cost)
            #import ipdb;
            #ipdb.set_trace()
            #start = times_init[0::2]
            #end = times_init[1::2]
        elif i == 1:
            #ref_traj, actual_traj, input_traj, cost_traj, times_opt = load_infbags(
            #    '/home/anusha/nn_final.bag',
            #    'dragonfly1', "/home/anusha/nn_poly_ref_times.pkl", 1)
            ref_traj, actual_traj, input_traj, cost_traj, times_opt = load_infbags('/home/anusha/replan/dragonfly1-poly.bag',
                                                                                    'dragonfly1',
                                                                                    "/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_poly.pkl",
                                                                                    0)
            total_cost = np.linalg.norm(np.vstack(ref_traj) - np.vstack(actual_traj))
            traj_costs.append(total_cost)
            #start = times_opt[0::2]
            #end = times_opt[1::2]
            # ref_traj, actual_traj, input_traj, cost_traj = load_infbags('/home/anusha/2023-03-01-07-59-27.bag', 'dragonfly1', 10)
        else:
            for rho in rhos:
                #ref_traj, actual_traj, input_traj, cost_traj, times = load_infbags('/home/anusha/kr.bag', 'dragonfly1', "/home/anusha/kr_times.pkl", 1)
                ref_traj, actual_traj, input_traj, cost_traj, times_poly = load_infbags(
                "/home/anusha/replan/dragonfly1-nn"+str(rho)+".bag",
                'dragonfly1',
                "/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_nn"+str(rho)+".pkl",
                rho)
                total_cost = np.linalg.norm(np.vstack(ref_traj) - np.vstack(actual_traj))
                traj_costs.append(total_cost)
            #start = times_poly[0::2]
            #end = times_poly[1::2]

        #cost = compute_tracking_cost(ref_traj, actual_traj, input_traj)
        #print(cost)

        #times = load_object("/home/anusha/nn_ref_times.pkl")

        #print(start)
        #print(end)

        #total_cost = compute_total_cost(start, end, np.array(cost))

        print("actual", actual_traj)

        #total_cost = np.linalg.norm(np.vstack(ref_traj) - np.vstack(actual_traj))
        #traj_costs.append(total_cost)
        """if i == 0:
            #traj_costs.append(compute_actual_cost(start, end, ref_traj, actual_traj, input_traj))
            cost = compute_tracking_cost(np.vstack(ref_traj), np.array(actual_traj), np.array(input_traj))
            total_cost = compute_total_cost(start, end, cost)
            traj_costs.append(total_cost)
        else:
            cost = compute_tracking_cost(np.vstack(ref_traj), np.array(actual_traj), np.array(input_traj))
            total_cost = compute_total_cost(start, end, cost)
            traj_costs.append(total_cost)"""

    rhos = [0, 1, 5, 10, 20, 50, 100]

    print("**********************************************")
    print(traj_costs)
    print("mj", np.sum(traj_costs[0]))
    print("poly", np.sum(traj_costs[1]))
    print("nn", np.sum(traj_costs[2]))

    #plt.scatter(np.array(rhos), np.ones(len(rhos)) * traj_costs[0], label="mj")
    plt.plot(np.array(rhos), np.ones(len(rhos)) * traj_costs[0], label="mj")
    #plt.scatter(np.array(rhos), np.ones(len(rhos)) * traj_costs[1], label="poly")
    plt.plot(np.array(rhos), np.ones(len(rhos)) * traj_costs[1], label="poly")
    plt.plot(np.array(rhos), np.array(traj_costs[2:]), label="ours")
    plt.scatter(np.array(rhos), np.array(traj_costs[2:]))
    plt.title("Tracking costs on replanned trajectories", fontsize=20)
    plt.xlabel("rho", fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel("Cost", fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc="center right", prop={'size':17})
    plt.savefig("/home/anusha/final_quad_plot_font.jpg", bbox_inches="tight")
    plt.show()

    #ours = np.reshape(traj_costs[1], [7, 10])
    #dev = np.std(ours, axis=1)


    plt.figure()

    # plt.scatter(np.array(rhos), np.ones(len(rhos)) * np.mean(traj_costs[2]), label="kr")
    plt.scatter(np.array(rhos), np.ones(len(rhos)) * np.mean(traj_costs[0]), label="poly")
    plt.scatter(np.array(rhos), np.mean(ours, axis=1), label="ours")
    plt.errorbar(np.array(rhos), np.mean(ours, axis=1), xerr=0, yerr=dev)
    plt.title("Tracking costs on 10 trajectories")
    plt.xlabel("rho")
    plt.ylabel("Cost")
    plt.legend(loc="center right")
    #plt.savefig("./main_data/final_quad_plot.png")
    plt.show()
    # save_object(traj_costs, '/home/anusha/Downloads/icra_results/cost_rho_mlp_post_icra.pkl')

    # Make box plots for each rho value as the final plot

    # For now just plot each rho using subplot
    """num_traj = 10
    fig, axes = plt.subplots(2, 1)
    for i in range(2):
        axes[i].plot(range(0, num_traj), traj_costs[i], 'r*', label='tracking cost')
        axes[i].set_title('Plot of cost for rho {:5.2f}, mean: {:6.2f}'.format(rho[i], np.mean(traj_costs[i])))
        axes[i].set_xlabel('Trajectory index')
        axes[i].set_ylabel('Total cost')
        axes[i].legend()
    k = 0
    for i in range(2):
        for j in range():
            axes[i, j].plot(range(0, num_traj), traj_costs[k], 'r*', label='tracking cost')
            axes[i, j].set_title('Plot of cost for rho {:5.2f}, mean: {:6.2f}'.format(rho[k], np.mean(traj_costs[k])))
            axes[i, j].set_xlabel('Trajectory index')
            axes[i, j].set_ylabel('Total cost')
            axes[i, j].legend()
            k += 1
    fig.savefig('/home/anusha/Downloads/plots2_new.png', bbox_inches='tight')
    plt.show()"""


if __name__ == '__main__':
    main()


