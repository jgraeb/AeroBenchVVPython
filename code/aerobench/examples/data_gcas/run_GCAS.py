'''
Modified code for Beyond-TMR project, original version by Stanley Bak

'''

import math
from numpy import deg2rad
import matplotlib.pyplot as plt
from aerobench.run_f16_sim import run_f16_sim, run_f16_sim_with_two_flight_computers, run_f16_sim_copy
from aerobench.visualize import plot
from aerobench.util import get_state_names, StateIndex
from gcas_autopilot import GcasAutopilot
from corrupted_gcas_autopilot import CorruptedGcasAutopilot
from ipdb import set_trace as st
import numpy as np
from mpl_toolkits import mplot3d

from mahalanobis import mahalanobis_distance

def set_initial_conditions(perturb = False):
    if perturb: # Gaussian for now - perturb each vector by random number from 1% std distribution
        mu, sigma = 0, 3 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 13)/100
    else:
        noise = np.zeros(13)

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1000        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = math.pi/8          # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = np.array([vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power])

    init = init + np.multiply(init,noise) # add the perturbations

    return init

def get_data(num_sims, tmax):
    results = []
    for i in range(0,num_sims):
        init = set_initial_conditions(perturb = True)
        # tmax = 3.51 # simulation time

        ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

        step = 1/30
        res = run_f16_sim(init, tmax, ap, step=step, integrator_str = 'rk45', extended_states=True)

        print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")
        results.append(res)
    return results

def run_corrupted_sim(tmax, bit_to_flip, flip_probability):
    print('Running corrupted simulation')
    init = set_initial_conditions(perturb = True)
    # tmax = 3.51 # simulation time

    cap = CorruptedGcasAutopilot(init_mode='roll', stdout=True, gain_str='old', bit_to_flip=bit_to_flip, flip_probability=flip_probability)

    step = 1/30
    res = run_f16_sim(init, tmax, cap, step=step, integrator_str='rk45',extended_states=True)
    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")

    return res

def run_2MR_sim(tmax, bit_to_flip, flip_probability, stuck_bit, time_of_corruption):
    print('Running corrupted simulation')
    init = set_initial_conditions(perturb = True)

    # Initialize flight computers
    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')
    cap = CorruptedGcasAutopilot(init_mode='roll', stdout=True, gain_str='old', bit_to_flip=bit_to_flip, flip_probability=flip_probability, stuck_bit=stuck_bit, time_of_corruption=time_of_corruption)
    # cap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')
    # cap = None
    step = 1/30
    res = run_f16_sim_with_two_flight_computers(init, tmax, ap, cap, step=step, integrator_str = 'rk45', extended_states=True, time_of_corruption=time_of_corruption)
    # res = run_f16_sim_with_two_flight_computers(init, tmax, ap, cap, step=step, integrator_str = 'rk45', extended_states=True)
    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")

    return res

def find_mean_and_variance(run_sim_result):
    times = []
    altitudes = []
    pos_es = []
    pos_ns = []
    us = []
    for res in run_sim_result:
        time = res['times']
        states = res['states']
        u = res['u']
        index_alt = get_state_names().index('alt')
        index_e = get_state_names().index('pos_e')
        index_n = get_state_names().index('pos_n')
        alts = states[:, index_alt] # 11: altitude (ft)
        es = states[:, index_e]
        ns = states[:, index_n]

        altitudes.append(alts)
        pos_es.append(es)
        pos_ns.append(ns)
        times.append(time)
        us.append(u)

    mean_pos = {}
    mean_pos.update({'pos_e': np.mean(pos_es, axis=0)})
    mean_pos.update({'pos_n': np.mean(pos_ns, axis=0)})
    mean_pos.update({'alt': np.mean(altitudes, axis=0)})
    mean_u = np.mean(us, axis=0)

    std_pos = {}
    std_pos.update({'pos_e': np.std(pos_es, axis=0)})
    std_pos.update({'pos_n': np.std(pos_ns, axis=0)})
    std_pos.update({'alt': np.std(altitudes, axis=0)})
    std_u = np.std(us, axis=0)



    return mean_pos, std_pos, mean_u, std_u

def plot_traj(run_sim_result, mean_pos, corrupted_results = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for res in run_sim_result:

        times = res['times']
        states = res['states']

        index_alt = get_state_names().index('alt')
        index_e = get_state_names().index('pos_e')
        index_n = get_state_names().index('pos_n')
        alts = states[:, index_alt] # 11: altitude (ft)
        es = states[:, index_e]
        ns = states[:, index_n]

        # ax.plot(times, ys, '-')
        ax.plot3D(es, ns, alts, 'red', alpha=0.05)

    ax.plot3D(mean_pos['pos_e'], mean_pos['pos_n'], mean_pos['alt'], 'red', linewidth=0.5)

    if corrupted_results:
        times = corrupted_results['times']
        cor_states = corrupted_results['states']
        cor_alts = cor_states[:, index_alt] # 11: altitude (ft)
        cor_es = cor_states[:, index_e]
        cor_ns = cor_states[:, index_n]
        ax.plot3D(cor_es, cor_ns, cor_alts, 'blue', linewidth=0.5)

    ax.set_ylabel('E-W')
    ax.set_xlabel('N-S')
    ax.view_init(elev=30, azim= 60)

    ax.set_title('Trajectories')
    plt.tight_layout()
    # plt.show()

    filename = 'trajectories.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_2MR_trajectories(sim_results, mean_pos, results):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for res in sim_results:

        # times = res['times']
        states = res['states']

        index_alt = get_state_names().index('alt')
        index_e = get_state_names().index('pos_e')
        index_n = get_state_names().index('pos_n')
        alts = states[:, index_alt] # 11: altitude (ft)
        es = states[:, index_e]
        ns = states[:, index_n]
        # ax.plot(times, ys, '-')
        ax.plot3D(es, ns, alts, 'red', alpha=0.05)

    ax.plot3D(mean_pos['pos_e'], mean_pos['pos_n'], mean_pos['alt'], 'red', linewidth=0.5)

    # times = results['times']
    num_vars = 16
    states = results['states']
    states_nom = [entry[:num_vars] for entry in states]
    states_cor = [entry[num_vars:] for entry in states]


    # states = results['states']
    # states_cap = results['states_cap']
    alts_orig = [entry[index_alt] for entry in states_nom] # 11: altitude (ft)
    es_orig = [entry[index_e] for entry in states_nom]#states[:, index_e]
    ns_orig = [entry[index_n] for entry in states_nom]#states[:, index_n]
    cor_alts = [entry[index_alt] for entry in states_cor]#states_cap[:, index_alt] # 11: altitude (ft)
    cor_es = [entry[index_e] for entry in states_cor]#states_cap[:, index_e]
    cor_ns = [entry[index_n] for entry in states_cor]#states_cap[:, index_n]
    # st()
    ax.plot3D(es_orig, ns_orig, alts_orig, 'red', linewidth=0.5)
    ax.plot3D(cor_es, cor_ns, cor_alts, 'blue', linewidth=0.5)

    ax.set_ylabel('E-W')
    ax.set_xlabel('N-S')
    ax.view_init(elev=30, azim= 60)

    ax.set_title('Trajectories')
    plt.tight_layout()

    # plt.show()

    filename = 'trajectories_2MR.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_control_signal(res, mean_u, std_u):
    # plot.init_plot()
    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(3, 1)
    # ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['u']
    states_cap = res['u_cap']
    # st()
    names = ['Throttle', 'Elevator', 'Aileron', 'Rudder']
    for i in range(1,4):
        ys = np.array(states)[:,i] # 11: altitude (ft)
        ax[i-1].plot(times, ys, '-', color = 'black', label=names[i], linewidth=1)
        ax[i-1].set_xlabel('Time [s]')
        ax[i-1].set_title(names[i])
        ax[i-1].grid(True,linestyle='--')
        ax[i-1].set_facecolor('whitesmoke')
        ax[i-1].set_xlim([0,times[-1]])
        ax[i-1].plot(times, np.array(mean_u)[:,i], '-', color = 'red', label=names[i], linewidth=1)
        ax[i-1].fill_between(times, np.array(mean_u)[:,i]-np.array(std_u)[:,i], np.array(mean_u)[:,i]+np.array(std_u)[:,i], color = 'red', alpha=0.2)
        # ax.plot(times, ys, '-', label=names[i])

    for i in range(1,4):
        ys = np.array(states_cap)[:,i] # 11: altitude (ft)
        ax[i-1].plot(times, ys, '-', color = 'blue', linewidth=1)

    # ax[i].set_ylabel('Control Signal')
    # ax.set_xlabel('Time')
    # ax.set_title('Value Control Signal')
    # plt.legend()
    plt.tight_layout()
    plt.legend()
    # st()

    filename = 'control_signal.pdf'
    # st()
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_control_signal_error(res):
    plot.init_plot()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['control_signal_error']
    control_strs = ['ctrl1', 'ctrl2', 'ctrl3', 'ctrl4']
    names = ['throttle', 'elevator (deg)', 'aileron (deg)', 'rudder (deg)']
    for i in range(4):
        ys = states[control_strs[i]] # 11: altitude (ft)
        ax.plot(range(len(ys)), ys, '.', label=names[i])

    ax.set_ylabel('Control Signal Error')
    ax.set_xlabel('Time')
    ax.set_title('Value Control Signal Error')
    plt.legend()
    plt.tight_layout()

    filename = 'control_signal_error.png'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_mahalanobis_distance_single_observation(data,obs):
    # prepare data
    times = []
    us = []

    for res in data:
        time = res['times']
        u = res['u']
        times.append(time)
        us.append(u)
    mean_u = np.mean(us, axis=0)
    std_u = np.std(us, axis=0)

    us = np.array(us)
    times = np.array(times[0]) # time step is the same

    # get control data from observation
    obs_time = np.array(res['times']) # same as res times
    obs_u_ap = np.array(obs['u'])

    mal_dist_ap = []
    for index in np.arange(len(times)):
        dist_ap = mahalanobis_distance(obs_u_ap[index,:], us[:,index,:])
        mal_dist_ap.append(dist_ap)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    ax.plot(times, mal_dist_ap, '-', color='red', linewidth=1, label='Uncorrupted  AP')

    ax.set_facecolor('whitesmoke')

    ax.set_ylabel('Distance Value')
    ax.set_xlabel('Time [s]')
    ax.set_title('Mahalanobis Distance')

    plt.tight_layout()
    plt.legend()
    plt.grid(True,linestyle='--')

    filename = 'mahalanobis_uncorrupted_2MR.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_pdf(data):
    # prepare data
    times = []
    us = []

    for res in data:
        time = res['times']
        u = res['u']
        times.append(time)
        us.append(u)
    mean_u = np.mean(us, axis=0)
    std_u = np.std(us, axis=0)

    us = np.array(us)
    times = np.array(times[0]) # time step is the same

    mean_dists = []
    max_dists = []
    all_dists = []
    for res in data:
        # u = res['u']
        dists = []
        # st()
        for index in np.arange(len(times)):
            dist = mahalanobis_distance(res['u'][index][:], us[:,index,:])
            dists.append(dist)
        all_dists.append(dists)
        # mean_dist = np.mean(dists)
        # max_dists.append(np.max(dists))
        # mean_dists.append(mean_dist)

    # st()
    all_dists_merged = []
    for entry in all_dists:
        all_dists_merged = all_dists_merged + entry
    # st()
    sorted_dists = np.sort(all_dists_merged)

    nums = []
    number_of_instances = 0
    for dist in sorted_dists:
        number_of_instances = number_of_instances + 1
        nums.append(number_of_instances/len(all_dists_merged))


    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    ax.plot(sorted_dists, nums, '-', color='red', linewidth=1)
    # ax.plot(sorted_dists, nums, '-', color='red', linewidth=1)

    ax.set_facecolor('whitesmoke')

    ax.set_ylabel('Fraction of Runs')
    ax.set_xlabel('Mahalanobis Distance')
    ax.set_title('Distribution')

    plt.tight_layout()
    plt.legend()
    plt.grid(True,linestyle='--')

    filename = 'mahal_2MR.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_mahalanobis_distance(data, obs):
    # prepare data
    times = []
    us = []

    for res in data:
        time = res['times']
        u = res['u']
        times.append(time)
        us.append(u)
    mean_u = np.mean(us, axis=0)
    std_u = np.std(us, axis=0)

    us = np.array(us)
    times = np.array(times[0]) # time step is the same

    # st()
    # get control data from observation
    obs_time = np.array(res['times']) # same as res times
    obs_u_ap = np.array(obs['u'])
    obs_u_cap = np.array(obs['u_cap'])

    mal_dist_ap = []
    mal_dist_cap = []
    # st()
    for index in np.arange(len(times)):
        dist_ap = mahalanobis_distance(obs_u_ap[index,:], us[:,index,:])
        dist_cap = mahalanobis_distance(obs_u_cap[index,:], us[:,index,:])
        mal_dist_ap.append(dist_ap)
        mal_dist_cap.append(dist_cap)
    # st()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    ax.plot(times, mal_dist_ap, '-', color='red', linewidth=1, label='Uncorrupted  AP')
    ax.plot(times, mal_dist_cap, '-', color='black', linewidth=1, label='Corrupted AP')

    ax.set_facecolor('whitesmoke')

    ax.set_ylabel('Distance Value')
    ax.set_xlabel('Time [s]')
    ax.set_title('Mahalanobis Distance')

    plt.tight_layout()
    plt.legend()
    plt.grid(True,linestyle='--')

    filename = 'mahal_2MR.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_2MR_altitudes(run_sim_result, mean_pos, result):
    # plot.init_plot()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    for res in run_sim_result:
        times = res['times']
        states = res['states']
        index = get_state_names().index('alt')
        ys = states[:, index] # 11: altitude (ft)
        ax.plot(times, ys, '-', color='red', alpha=0.05)

    ax.plot(times, mean_pos['alt'], '-', color='red', linewidth=0.5)

    # st()
    num_vars = 16
    states = result['states']
    states_nom = [entry[:num_vars] for entry in result['states']]
    states_cor = [entry[num_vars:] for entry in result['states']]
    times_cor = result['times']
    index = get_state_names().index('alt')
    corrupted_alt = [entry[index] for entry in states_cor] # 11: altitude (ft)
    ax.plot(times_cor, corrupted_alt, '-.', color='blue', linewidth=0.5, label='Corrupted OBC')

    index = get_state_names().index('alt')
    nom_alt = [entry[index] for entry in states_nom] # 11: altitude (ft)
    ax.plot(times_cor, nom_alt, '-.', color='red', linewidth=0.5, label='Nominal OBC')

    ax.set_facecolor('whitesmoke')

    ax.set_ylabel('Altitude [ft]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Altitude')

    plt.tight_layout()
    plt.legend()
    plt.grid(True,linestyle='--')

    filename = 'altitudes_2MR.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def plot_altitudes(run_sim_result, mean_pos, corrupted_results = None):
    # plot.init_plot()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    for res in run_sim_result:
        times = res['times']
        states = res['states']
        index = get_state_names().index('alt')
        ys = states[:, index] # 11: altitude (ft)
        ax.plot(times, ys, '-', color='red', alpha=0.02)

    ax.plot(times, mean_pos['alt'], '-', color='red', linewidth=0.5)
    if corrupted_results:
        states = corrupted_results['states']
        index = get_state_names().index('alt')
        corrupted_alt = states[:, index] # 11: altitude (ft)
        ax.plot(times, corrupted_alt, '-', color='blue', linewidth=0.5)

    ax.set_ylabel('Altitude')
    ax.set_xlabel('Time')
    ax.set_title('Altitude (ft)')
    plt.tight_layout()

    filename = 'altitudes.pdf'
    plt.savefig(filename)
    print(f"Made {filename}")

def postprocess(results, corrupted_results):
    mean_pos, std_pos, mean_u, std_u = find_mean_and_variance(results)
    plot_altitudes(results, mean_pos, corrupted_results)
    plot_traj(results, mean_pos, corrupted_results)
    plot_control_signal_error(corrupted_results)
    plt.show()

def compare_2MR(nom_results, res):
    mean_pos, std_pos, mean_u, std_u = find_mean_and_variance(nom_results)
    # st()
    plot_control_signal(res, mean_u, std_u)
    plot_2MR_altitudes(nom_results, mean_pos, res)
    plot_2MR_trajectories(nom_results, mean_pos, res)

    plot_mahalanobis_distance(nom_results, res)

    # plot_mahalanobis_distance_single_observation(nom_results,res)
    # plot_altitudes(nom_results, mean_pos)
    # plot_traj(nom_results, mean_pos)
    # plot_control_signal(res, mean_u, std_u)
    plot_pdf(nom_results)

    # plot_2MR_traj(results, mean_pos, corrupted_results)
    # plot_control_signal_error(res)
    # plot_control_signal(res)
    plt.show()

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1000        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = -math.pi/8           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 3.51 # simulation time

    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

    step = 1/30
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)

    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    num_sims = 50
    tmax = 4
    bit_to_flip = 2
    flip_probability = 1.0
    stuck_bit = True
    time_of_corruption = 1.0
    nom_results = get_data(num_sims, tmax)
    # corrupted_results = run_corrupted_sim(tmax, bit_to_flip, flip_probability)
    # st()
    res = run_2MR_sim(tmax, bit_to_flip, flip_probability, stuck_bit, time_of_corruption)
    compare_2MR(nom_results, res)
    # postprocess(results, corrupted_results)
