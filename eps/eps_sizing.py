import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import global_constants
from global_constants import TOGA # Import global constants

# P_req = 500000  #[W] Required power in watts
# Cable_length = 20 #Cable length in meter
# Spec_W_cable = 0.0015 # cables: 0.0015 kg/kW/m

def motor_weight(P_req):
    """Calculate the weight of the EPS based on the required power."""
    return P_req / 10 / 1000

def inverter_weight(P_req):
    return 0.97 * P_req /20 / 1000

def cable_weight(P_req, cable_length):
    Spec_W_cable = 0.0015  # kg/kW/m, specific weight of the cable
    return cable_length * Spec_W_cable * P_req / 1000

def heat_dissipation_motor(P_req):
    """Calculate the heat dissipation of the EPS."""
    return P_req * (1/0.97 - 1)

def heat_dissipation_inverter(P_req):
    """Calculate the heat dissipation of the EPS."""
    return P_req * (1/0.99 - 1)

def fc_power_needed(P_req):
    """Calculate the total power needed from the fuel cell."""
    return P_req / (0.97*0.99)

def motor_sizing(P_req):
    """Calculate the EPS sizing parameters."""
    lambd = 0.5  # [m] Length of the motor
    c1 = 0.000805
    c2 = 0.00161
    c3 = 0.1934
    hs = 0.025
    P_req2 = P_req / 2
    r = (lambd * P_req2)**(1/3) * c1
    l = c2 * 1/lambd * (lambd * P_req2)**(1/3)
    hy = r * c3
    r_out = r + hs + hy
    r_tot = ((0.737 * (r_out ** 2)) - 0.580 * r_out + 1.1599) * r_out
    return r_tot, l

def motor_volume(radius, length):
    """Calculate the volume of the EPS."""
    return np.pi * radius**2 * length

def inverter_volume(P_req):
    return 0.97 * P_req / 18.7

# print("Motors weight:",motor_weight(P_req),"Inverter Weight:", inverter_weight(P_req),"Cable weight:", cable_weight(P_req, Cable_length) )
# print("1 Motor radius and length:",motor_sizing(0.5, P_req))

def eps_main(powersplit):
    P_req = TOGA * powersplit  # Required power in watts, scaled by the power split factor
    """Main function to calculate EPS parameters."""
    Cable_length = 0 # Modify this value as needed
    # Calculate weights
    motor_w = motor_weight(P_req)
    inverter_w = inverter_weight(P_req)
    cable_w = cable_weight(P_req, Cable_length)

    # Calculate heat dissipation
    heat_motor = heat_dissipation_motor(P_req)
    heat_inverter = heat_dissipation_inverter(P_req)

    # Calculate fuel cell power needed
    fc_power = fc_power_needed(P_req)

    # Motor sizing
    radius, length = motor_sizing(P_req)
    
    # Calculate volumes
    motor_vol = motor_volume(radius, length)
    inverter_vol = inverter_volume(P_req)

    co2_per_kg = 40
    motor_co2 = motor_w * co2_per_kg

    return (motor_w + inverter_w + cable_w), (heat_motor + heat_inverter), motor_vol, motor_co2, P_req

#print(eps_main(0.5))  