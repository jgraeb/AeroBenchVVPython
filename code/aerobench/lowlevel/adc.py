'''
Stanley Bak
adc.py for F-16 model
'''

from math import sqrt
from ipdb import set_trace as st
import numpy as np

def adc(vt, alt):
    '''converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    '''
    perturb = True
    noise = [0,0]

    if perturb:
        noise[0] = np.random.normal(0,10,1) / 100 # density noise in percent
        noise[1] = np.random.normal(0,10,1) / 100 # temperature noise in percent

    # vt = freestream air speed

    ro = 2.377e-3 + 2.377e-3 * noise[0][0]  # slugs/f^3
    tfac = 1 - .703e-5 * alt # from sea level to altitude

    if alt >= 35000: # in stratosphere constant temperature
        t = 390
    else: # in troposphere
        t_f_true = 59 # standard sea level temperatue in Fahrenheit (15 C)
        t_f = t_f_true + t_f_true * noise[1][0] # add noise to Fahrenheit value
        t_r = t_f + 459.67 # convert to Rankine
        t = (t_r) * tfac # 3 rankine per atmosphere (3 rankine per 1000 ft)

    # rho = freestream mass density
    rho = ro * tfac**4.14

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a

    # qbar = dynamic pressure
    qbar = .5 * rho * vt * vt
    # st()
    return amach, qbar
