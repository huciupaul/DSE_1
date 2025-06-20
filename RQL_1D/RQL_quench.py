import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))

import cantera as ct
ct.suppress_thermo_warnings()

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
from scipy.interpolate import RegularGridInterpolator, InterpolatedUnivariateSpline
from matplotlib.colors import ListedColormap
import json

from MECH import MECH
from mixture_properties import mixture_properties_with_temp

'''
This script is used to estimate the NOx formation in the quench zone of a RQL combustor.
'''

# Input gases
class gas_obj:
    mdot: float # kg/s
    X: str      # Cantera mole fraction string
    T: float    # K
    P: float    # Pa
    def __init__(self, mdot: float, X: str, T: float, P: float):
        self.mdot = mdot
        self.X = X
        self.T = T
        self.P = P

        self.gas = ct.Solution(MECH)
        self.gas.TPX = T, P, X


def main(
        power_arr: np.ndarray,
        mdot_h2o: np.ndarray,
        water_liquid: bool = True
        ) -> Tuple[List[float], np.ndarray]:


    with open(r'C:\Users\zksie\OneDrive - Delft University of Technology\Documents\Academic Year 2024-2025\Q4 - DSE\Emissions_tool\DSE_1\RQL_1D\rich_cases.json', 'r') as f:
        data = json.load(f)
        
    cases = data["cases"]
    power_plotted = np.array([case['power_kW'] for case in cases])
    P_const = np.array([case['P_inlet_Pa'] for case in cases])
    T_rich_exh = np.array([case['T_peak_K'] for case in cases])
    mdot_rich_exh = np.array([case['mdot_exh'] for case in cases])
    X_in = np.array([case['X_exh'] for case in cases])

    P_spline = InterpolatedUnivariateSpline(power_plotted, P_const, k=3)
    T_spline = InterpolatedUnivariateSpline(power_plotted, T_rich_exh, k=3)
    mdot_exh_spline = InterpolatedUnivariateSpline(power_plotted, mdot_rich_exh, k=3)
    
    P_out = P_spline(power_arr)
    T_out = T_spline(power_arr)
    mdot_exh = mdot_exh_spline(power_arr)


    NOx_normalized = np.full((len(power_arr), len(mdot_h2o)), np.nan)
    NOx_at_autoignition = np.full((len(power_arr), len(mdot_h2o)), np.nan)
    P_matrix = np.zeros((len(power_arr), len(mdot_h2o)))
    time_max_dT_dlogt = np.zeros((len(power_arr), len(mdot_h2o)))
    res_time = np.zeros((len(power_arr), len(mdot_h2o)))


    for i in tqdm(range(len(power_arr))):
        idx_closest_power = np.argmin(np.abs(power_plotted - power_arr[i]))
        X_in_i = X_in[idx_closest_power]
        # Exhaust of the rich zone
        rich_ex = gas_obj(
                mdot=mdot_exh[i],
                X=X_in_i,
                T= T_out[i],
                P= P_out[i]
                )

        # Air bled, to be mixed with rich exhaust to create stoichiometric mixture
        air = gas_obj(
                mdot=1.0,
                X="N2:0.78084, O2:0.20946",
                T=603.0,
                P=rich_ex.P
                )

        mdot_h2 = rich_ex.mdot * rich_ex.gas.X[1]
        mdot_o2 = 0.5 * mdot_h2  # stoichiometric ratio for H2 combustion, assuming no O2 in exhaust
        air.mdot = mdot_o2 / air.gas.X[3]

        mdot_mix1, T_mix1, P_mix1, X_mix1 = mixture_properties_with_temp(
            stream1=rich_ex,
            stream2=air,
            MECH=MECH
        )

        stoich_mix = gas_obj(
                mdot=mdot_mix1,
                X=X_mix1,
                T=T_mix1,
                P=P_mix1
                )
        
        for j, m_dot_h2o_j in enumerate(mdot_h2o):
            # Liquid water injection
            h2o = gas_obj(
                    mdot=m_dot_h2o_j,
                    X="H2O:1",
                    T=353.0,
                    P=rich_ex.P
                    )
            
            if water_liquid:
                vapor_h = h2o.gas.enthalpy_mass
                del_h = 2580 * 1e3 # J/kg, simple estimation
                h2o.HP = (vapor_h - del_h), h2o.gas.P
            

            mdot_mix, T_mix, P_mix, X_mix = mixture_properties_with_temp(
                stream1=stoich_mix,
                stream2=h2o,
                MECH=MECH
            )

            stoich_with_water = gas_obj(
                mdot=mdot_mix,
                X=X_mix,
                T=T_mix,
                P=P_mix
            )
            
            # Set up the Cantera gas object
            reactor = ct.IdealGasConstPressureReactor(stoich_with_water.gas)
            sim = ct.ReactorNet([reactor])

            Temp = []
            X_nox = []
            time = []

            V = stoich_with_water.mdot / (2*np.pi*0.15*0.02 * stoich_with_water.gas.density)
            tau_max = 0.1/V
            res_time[i, j] = tau_max

            for t in np.arange(1e-9, tau_max, 1e-6):
                try:
                    sim.advance(t)
                    Temp.append(reactor.thermo.T)
                    X_nox.append(reactor.thermo['NO'].X[0])
                    time.append(t)

                except Exception as e:
                    print("Error at m_dot_h2o_i =", m_dot_h2o_j)
                    print("T =", stoich_with_water.gas.T)
                    print("P =", stoich_with_water.gas.P)
                    print("X =", stoich_with_water.gas.X)
                
                    if reactor.thermo.T > 20000 or reactor.thermo.T < 200:
                        print(f"Unphysical T = {reactor.thermo.T}, skipping this iteration.")
                        print(f"Reactor state before failure:\nT = {reactor.thermo.T}, Y = {reactor.thermo.Y}")
                        print(f"Sum(Y) = {sum(reactor.thermo.Y)}")
                        raise

            log_time = np.log10(time)
            dT_dlogt = np.diff(Temp) / np.diff(log_time)
            time_mid = 10**((log_time[:-1] + log_time[1:]) / 2)
            idx_max = np.argmax(dT_dlogt)

            # plt.figure()
            # plt.plot(time_mid, dT_dlogt)
            # plt.xscale('log')
            # plt.xlabel('Time [s] (log scale)')
            # plt.ylabel('dT/d(log t)')
            # plt.title('Derivative of Temperature vs Log(Time)')
            # plt.show()
            
            # plt.figure()
            # plt.plot(np.arange(1e-9, tau_max, 1e-6), Temp)
            # plt.xscale('log')
            # plt.show()            

            # X_NOx[i, j] = reactor.thermo['NO'].X[0] # molar fraction # * stoich_with_water.mdot # kg/s
            # X_O2[i, j] = reactor.thermo['O2'].X[0]
            # X_H2O[i, j] = reactor.thermo['H2O'].X[0]

            P_matrix[i, j] = reactor.thermo.P
            time_max_dT_dlogt[i, j] = time_mid[idx_max]

            O2_dry = reactor.thermo['O2'].X[0] / (1.0 - reactor.thermo['H2O'].X[0])
            F15 = (20.9 - 15.0) / (20.9 - 100.0 * O2_dry)
            
            NOx_normalized[i, j] = reactor.thermo['NO'].X[0] * F15 #* 1e6
            NOx_at_autoignition[i, j] = X_nox[idx_max] * F15 #* 1e6

    return NOx_normalized, NOx_at_autoignition, P_matrix, time_max_dT_dlogt, res_time


if __name__ == "__main__":
    power_arr = np.arange(64.5, 1223, 10)
    #power_arr = [64.5, 813, 1223]
    mdot_h2o = np.arange(0, 0.1 + 1e-2, 1e-2)

    NOx_normalized, NOx_at_autoignition, P_matrix, time_max_dT_dlogt, res_time = main(power_arr=power_arr, mdot_h2o=mdot_h2o, water_liquid=True)

    T_grid, W_grid = np.meshgrid(power_arr, mdot_h2o, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W_grid, T_grid, NOx_normalized, cmap='viridis')

    # Add 25 ppm contour (2.5e-5)
    contour = ax.contour(
        W_grid, T_grid, NOx_normalized,
        levels=[2.5e-5],
        colors='red',
        linewidths=2,
        offset=2.5e-5  # Place contour at the correct z-level
    )
    ax.clabel(contour, fmt={2.5e-5: '25 ppm'}, colors='red')

    ax.set_xlabel('Water injection [kg/s]')
    ax.set_ylabel('Power [kW]')
    ax.set_zlabel('NOx formed [kg/s]')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()


    plt.figure()
    # Show heatmap: x-axis = water injection, y-axis = temperature
    plt.imshow(
        res_time.T,  # Transpose if you want x=W, y=T
        origin='lower',
        aspect='auto',
        extent=[W_grid.min(), W_grid.max(), T_grid.min(), T_grid.max()],
        cmap='viridis'
    )
    cbar = plt.colorbar(label='Residence time [s]')

    # Set ticks and labels
    yticks = np.linspace(T_grid.min(), T_grid.max(), num=6)
    plt.yticks(yticks)
    plt.xlabel('Water injection [kg/s]')
    plt.ylabel('Power [kW]')
    plt.show()


    plt.figure()
    # Show heatmap: x-axis = water injection, y-axis = temperature
    plt.imshow(
        time_max_dT_dlogt.T,  # Transpose if you want x=W, y=T
        origin='lower',
        aspect='auto',
        extent=[W_grid.min(), W_grid.max(), T_grid.min(), T_grid.max()],
        cmap='viridis'
    )
    cbar = plt.colorbar(label='Autoignition time [s]')

    # Set ticks and labels
    yticks = np.linspace(T_grid.min(), T_grid.max(), num=6)
    plt.yticks(yticks)
    plt.xlabel('Water injection [kg/s]')
    plt.ylabel('Power [kW]')
    plt.show()


    # Create mask: True (green) if res_time < time_max_dT_dlogt, else False (red)
    mask = (res_time < time_max_dT_dlogt).T  # Transpose to match your plotting

    # Define custom colormap: green for True, red for False
    cmap = ListedColormap(['red', 'green'])

    plt.figure()
    plt.imshow(
        mask,
        origin='lower',
        aspect='auto',
        extent=[W_grid.min(), W_grid.max(), T_grid.min(), T_grid.max()],
        cmap=cmap
    )
    plt.xlabel('Water injection [kg/s]')
    plt.ylabel('Power [kW]')
    plt.show()

