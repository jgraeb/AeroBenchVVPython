'''
autopilot modified from GcasAutopilot with added bitflips.
'''

import math
import numpy as np
from numpy import deg2rad
from aerobench.highlevel.autopilot import Autopilot
from aerobench.util import StateIndex
from aerobench.lowlevel.low_level_controller import LowLevelController
from gcas_autopilot import GcasAutopilot
from ipdb import set_trace as st
import struct
from copy import deepcopy
import random

def float_to_bin(num):
    the_str = '0b'
    return the_str + bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def make_str(bit_to_flip):
    the_str = '0b'
    for i in range(32):
        if i == bit_to_flip:
            the_str = the_str + str(1)
        else:
            the_str = the_str + str(0)
    return the_str

def flip_bit_in_bin(bin_full, bit_to_flip):
    bin = bin_full[2:]
    the_str = '0b'
    for i in range(32):
        if i == bit_to_flip:
            if bin[i] == '0':
                the_str = the_str + str(1)
            else:
                the_str = the_str + str(0)
        else:
            the_str = the_str + bin[i]
    return the_str

def flip_bit_in_float(num, bit_to_flip, range): # 32 bit float number and flip bit
    bin_to_flip = float_to_bin(num)
    print(bin_to_flip)
    flipped_bin = flip_bit_in_bin(bin_to_flip, bit_to_flip)
    print(flipped_bin)
    flipped_float = bin_to_float(flipped_bin)
    # saturate
    if flipped_float > range[1]:
        flipped_float = range[1]
    elif flipped_float < range[0]:
        flipped_float = range[0]
    if np.isnan(flipped_float): # set value to 0.0 if flipped float is NaN
        flipped_float = 0.0

    print("corrupted {0} to {1} by flipping of bit {2}".format(num, flipped_float, bit_to_flip))
    return flipped_float

def flip_bit(int_to_flip, min_range, max_range, mask, bit_to_flip): # scale 64 bit number and flip bit
    max_int = 18446744073709551615 # max limit for 64 bit integer
    scaled_int = int_to_flip/max_range * max_int
    scaled_flipped_bin = bin(int(scaled_int) ^ mask)
    # print(scaled_flipped_bin)
    flipped_int = int(scaled_flipped_bin, 2) / max_int * max_range
    print("corrupted {0} to {1} by flipping of bit {2}".format(int_to_flip, flipped_int, bit_to_flip))
    return flipped_int

def get_control_signal_error(rv, rv_cor):
    delta_control = np.zeros(4)
    # st()
    for i in range(len(rv)):
        delta_control[i] = (rv_cor[i]-rv[i])
    return delta_control

class CorruptedGcasAutopilot(GcasAutopilot):
    def __init__(self, init_mode='standby', gain_str='old', stdout=False, bit_to_flip=34, flip_probability = 0.1):

        assert init_mode in ['standby', 'roll', 'pull', 'waiting']
        # added for corrupting computation results
        self.bit_to_flip = bit_to_flip
        self.mask = int(make_str(bit_to_flip), 2) # bit location to flip
        self.control_signal_error_dict = {'ctrl1': [], 'ctrl2': [], 'ctrl3': [], 'ctrl4': []}
        self.flip_probability = flip_probability # % of cases the bit will be flipped

        # config
        self.cfg_eps_phi = deg2rad(5)       # Max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # Max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # Min path angle before completion
        self.cfg_k_prop = 4                 # Proportional control gain
        self.cfg_k_der = 2                  # Derivative control gain
        self.cfg_flight_deck = 1000         # Altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # Min duration of pull up

        self.cfg_nz_des = 5

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = np.zeros(4)
        self.waiting_time = 2

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, init_mode, llc=llc)

    def get_u_ref(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode == 'standby':
            rv = np.zeros(4)
        elif self.mode == 'waiting':
            rv = self.waiting_cmd
        elif self.mode == 'roll':
            rv = self.roll_wings_level(x_f16)
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"
            rv = self.pull_nose_level()

        # make sure rv stays in bounds
        # for i,entry in enumerate(rv):
        #     if entry > 6:
        #         rv[i] = 6
        #     elif entry < -1:
        #         rv[i] = -1

        rv_cor = deepcopy(rv) # corrupt the data
        if random.random() < self.flip_probability:
            # for i in range(len(rv)):
            #     # rv[i] = flip_bit(rv[i],0,6,self.mask, self.bit_to_flip)
            #     rv_cor[i] = flip_bit_in_float(rv[i], self.bit_to_flip, [-1,6])
            rv_cor[1] = flip_bit_in_float(rv[1], self.bit_to_flip, [-1,6])
        # st()
        delta_control = get_control_signal_error(rv, rv_cor)
        # print('The control error is: {}'.format(delta_control))
        self.control_signal_error_dict['ctrl1'].append(delta_control[0])
        self.control_signal_error_dict['ctrl2'].append(delta_control[1])
        self.control_signal_error_dict['ctrl3'].append(delta_control[2])
        self.control_signal_error_dict['ctrl4'].append(delta_control[3])
        return rv_cor


    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        premode = self.mode

        if self.mode == 'waiting':
            # time-triggered start after two seconds
            if t + 1e-6 >= self.waiting_time:
                self.mode = 'roll'
        elif self.mode == 'standby':
            if not self.is_nose_high_enough(x_f16) and not self.is_above_flight_deck(x_f16):
                self.mode = 'roll'
        elif self.mode == 'roll':
            if self.is_roll_rate_low(x_f16) and self.are_wings_level(x_f16):
                self.mode = 'pull'
                self.pull_start_time = t
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"

            if self.is_nose_high_enough(x_f16) and t >= self.pull_start_time + self.cfg_min_pull_time:
                self.mode = 'standby'

        rv = premode != self.mode

        if rv:
            self.log(f"GCAS transition {premode} -> {self.mode} at time {t}")

        return rv
