'''
Stanley Bak
run_f16_sim python version
'''

import time
import numpy as np
from scipy.integrate import RK45
from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler
from ipdb import set_trace as st
from aerobench.util import StateIndex

def run_f16_sim_with_two_flight_computers(initial_state, tmax, ap, cap, step=1/30, extended_states=False, model_str='morelli',
                integrator_str='rk45', v2_integrators=False):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    start = time.perf_counter()

    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc
    # llc_cap = cap.llc
    # st()
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    # st()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states_cap = [x0]
    states_ap = [x0]
    states = [np.concatenate((states_ap[0], states_cap[0]), axis=0)] # integrate both at the same time

    # mode can change at time 0
    ap.advance_discrete_mode(times[-1], states_ap[-1])
    cap.advance_discrete_mode(times[-1], states_cap[-1])

    modes = [ap.mode]
    modes_cap = [cap.mode]
    assert ap.mode == cap.mode # first state is the same


    if extended_states:
        xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states_ap[-1], model_str, v2_integrators)

        xd_list = [xd]
        u_list = [u]
        Nz_list = [Nz]
        ps_list = [ps]
        Ny_r_list = [Ny_r]

        xd_cap, u_cap, Nz_cap, ps_cap, Ny_r_cap = get_extended_states(cap, times[-1], states_cap[-1], model_str, v2_integrators)

        xd_list_cap = [xd_cap]
        u_list_cap = [u_cap]
        Nz_list_cap = [Nz_cap]
        ps_list_cap = [ps_cap]
        Ny_r_list_cap = [Ny_r_cap]

    der_func = make_der_func_auxiliary(ap, cap, model_str, v2_integrators)
    # st()
    # der_func = make_der_func(ap, model_str, v2_integrators)
    # der_func_cap = make_der_func_2(cap, model_str, v2_integrators)
    # der_func_cap = der_func

    if integrator_str == 'rk45':
        integrator_class = RK45
        integrator_class_cap = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        integrator_class_cap = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
    # integrator_cap = integrator_class_cap(der_func, times[-1], states_cap[-1], tmax, **kwargs)


    while integrator.status == 'running':# and integrator_cap.status == 'running':
        # integrator.h_abs = step
        # integrator_cap.h_abs = step
        integrator.step()
        # integrator_cap.step()

        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()
            # dense_output_cap = integrator_cap.dense_output()


            while integrator.t >= times[-1] + step:
                # st()
                t = times[-1] + step
                #print(f"{round(t, 2)} / {tmax}")
                # st()
                times.append(t)
                states.append(dense_output(t))
                # states_cap.append(dense_output_cap(t))

                # if np.abs(states[-1][11]-states_cap[-1][11]) > 50:
                #     print(states[-1][11])
                #     print(states_cap[-1][11])
                #     # st()

                updated = ap.advance_discrete_mode(times[-1], states[-1][:num_vars])
                modes.append(ap.mode)

                updated_cap = cap.advance_discrete_mode(times[-1], states[-1][num_vars:])
                modes_cap.append(cap.mode)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1][:num_vars], model_str, v2_integrators)

                    xd_list.append(xd)
                    u_list.append(u)
                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    Ny_r_list.append(Ny_r)

                    xd_cap, u_cap, Nz_cap, ps_cap, Ny_r_cap = get_extended_states(cap, times[-1], states[-1][num_vars:], model_str, v2_integrators)

                    xd_list_cap.append(xd_cap)
                    u_list_cap.append(u_cap)
                    Nz_list_cap.append(Nz_cap)
                    ps_list_cap.append(ps_cap)
                    Ny_r_list_cap.append(Ny_r_cap)

                if ap.mode == 'standby' and cap.mode == 'standby':
                    break

                if ap.is_finished(times[-1], states[-1][:num_vars]):# and cap.is_finished(times[-1], states_cap[-1]):
                    # this both causes the outer loop to exit and sets res['status'] appropriately
                    integrator.status = 'autopilot finished'
                    integrator_cap.status = 'autopilot finished'
                    st()
                    break

                breaking = False

                if updated or updated_cap:
                    # re-initialize the integration class on discrete mode switches
                    # st()
                    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    # integrator_cap = integrator_class(der_func_cap, times[-1], states_cap[-1], tmax, **kwargs)
                    breaking = True

                # if updated_cap:
                #     # st()
                #     # re-initialize the integration class on discrete mode switches
                #     integrator_cap = integrator_class_cap(der_func_cap, times[-1], states_cap[-1], tmax, **kwargs)
                #     breaking = True

                if breaking:
                    break
    # st()
    # assert 'finished' in integrator.status

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes

    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        res['u_list'] = u_list

    res['runtime'] = time.perf_counter() - start

    res['control_signal_error'] = cap.control_signal_error_dict

    # res['status_cap'] = integrator_cap.status
    res['states_cap'] = np.array(states_cap, dtype=float)
    res['modes_cap'] = modes_cap
    res['u'] = u_list
    res['u_cap'] = u_list_cap

    return res

def run_f16_sim(initial_state, tmax, ap, step=1/30, extended_states=False, model_str='morelli', integrator_str='rk45', v2_integrators=False):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    start = time.perf_counter()

    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc

    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states = [x0]

    # mode can change at time 0
    ap.advance_discrete_mode(times[-1], states[-1])

    modes = [ap.mode]

    if extended_states:
        xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

        xd_list = [xd]
        u_list = [u]
        Nz_list = [Nz]
        ps_list = [ps]
        Ny_r_list = [Ny_r]

    der_func = make_der_func(ap, model_str, v2_integrators)

    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)

    while integrator.status == 'running':
        integrator.step()

        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()

            while integrator.t >= times[-1] + step:
                t = times[-1] + step
                #print(f"{round(t, 2)} / {tmax}")

                times.append(t)
                states.append(dense_output(t))

                updated = ap.advance_discrete_mode(times[-1], states[-1])
                modes.append(ap.mode)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

                    xd_list.append(xd)
                    u_list.append(u)

                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    Ny_r_list.append(Ny_r)

                if ap.is_finished(times[-1], states[-1]):
                    # this both causes the outer loop to exit and sets res['status'] appropriately
                    integrator.status = 'autopilot finished'
                    break

                if updated:
                    # re-initialize the integration class on discrete mode switches
                    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    break

    assert 'finished' in integrator.status

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes

    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        res['u_list'] = u_list

    res['runtime'] = time.perf_counter() - start

    res['control_signal_error'] = ap.control_signal_error_dict
    res['u'] = u_list

    return res


def run_f16_sim_copy(initial_state, tmax, ap, cap, step=1/30, extended_states=False, model_str='morelli',
                integrator_str='rk45', v2_integrators=False):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    start = time.perf_counter()

    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc

    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states = [x0]
    states_cap = [x0]

    # mode can change at time 0
    ap.advance_discrete_mode(times[-1], states[-1])
    cap.advance_discrete_mode(times[-1], states_cap[-1])

    modes = [ap.mode]
    modes_cap = [cap.mode]

    if extended_states:
        xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

        xd_list = [xd]
        u_list = [u]
        Nz_list = [Nz]
        ps_list = [ps]
        Ny_r_list = [Ny_r]

        xd_cap, u_cap, Nz_cap, ps_cap, Ny_r_cap = get_extended_states(cap, times[-1], states_cap[-1], model_str, v2_integrators)

        xd_list_cap = [xd_cap]
        u_list_cap = [u_cap]
        Nz_list_cap = [Nz_cap]
        ps_list_cap = [ps_cap]
        Ny_r_list_cap = [Ny_r_cap]

    der_func = make_der_func(ap, model_str, v2_integrators)
    der_func_cap = make_der_func(cap, model_str, v2_integrators)

    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
    integrator_cap = integrator_class(der_func_cap, times[-1], states_cap[-1], tmax, **kwargs)

    while integrator.status == 'running':# and integrator_cap == 'running':
        try:
            integrator.step()
            integrator_cap.step()
        except:
            st()

        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()
            dense_output_cap = integrator_cap.dense_output()

            while integrator.t >= times[-1] + step:
                t = times[-1] + step
                #print(f"{round(t, 2)} / {tmax}")
                st()
                times.append(t)
                states.append(dense_output(t))
                states_cap.append(dense_output_cap(t))

                updated = ap.advance_discrete_mode(times[-1], states[-1])
                updated_cap = cap.advance_discrete_mode(times[-1], states_cap[-1])
                modes.append(ap.mode)
                modes_cap.append(cap.mode)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

                    xd_list.append(xd)
                    u_list.append(u)

                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    Ny_r_list.append(Ny_r)

                    xd_cap, u_cap, Nz_cap, ps_cap, Ny_r_cap = get_extended_states(cap, times[-1], states_cap[-1], model_str, v2_integrators)

                    xd_list_cap.append(xd_cap)
                    u_list_cap.append(u_cap)
                    Nz_list_cap.append(Nz_cap)
                    ps_list_cap.append(ps_cap)
                    Ny_r_list_cap.append(Ny_r_cap)

                if ap.is_finished(times[-1], states[-1]) or cap.is_finished(times[-1], states_cap[-1]):
                    # this both causes the outer loop to exit and sets res['status'] appropriately
                    integrator.status = 'autopilot finished'
                    integrator_cap.status = 'autopilot finished'
                    break

                if updated or updated_cap:
                    if updated:
                        # re-initialize the integration class on discrete mode switches
                        integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    if updated_cap:
                        integrator_cap = integrator_class(der_func_cap, times[-1], states_cap[-1], tmax, **kwargs)

                    break

    # assert 'finished' in integrator.status

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes

    res['states_cap'] = np.array(states_cap, dtype=float)
    res['modes_cap'] = modes_cap

    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        res['u_list'] = u_list

        res['xd_list_cap'] = xd_list_cap
        res['ps_list_cap'] = ps_list_cap
        res['Nz_list_cap'] = Nz_list_cap
        res['Ny_r_list_cap'] = Ny_r_list_cap
        res['u_list_cap'] = u_list_cap

    res['runtime'] = time.perf_counter() - start

    res['control_signal_error'] = cap.control_signal_error_dict
    res['u'] = u_list
    res['u_cap'] = u_list_cap

    return res

def make_der_func_auxiliary(ap, cap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)
        u_refs_cap = cap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = (len(get_state_names()) + ap.llc.get_num_integrators())
        # assert full_state.size // num_vars == num_aircraft

        xds = []

        # for i in range(num_aircraft):
        state = full_state[0:num_vars]
        u_ref = u_refs[0:4]

        state_cap = full_state[num_vars:2*num_vars+1]
        u_ref_cap = u_refs_cap[0:4]

        xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
        xd_cap = controlled_f16(t, state_cap, u_ref_cap, cap.llc, model_str, v2_integrators)[0]

        xds.append(xd)
        xds.append(xd_cap)

        rv = np.hstack(xds)
        # st()
        return rv

    return der_func

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func

def make_der_func_2(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func_2(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func_2

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r
