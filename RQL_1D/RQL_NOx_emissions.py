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
from typing import List, Tuple, Dict
import pickle
from scipy.interpolate import RegularGridInterpolator, InterpolatedUnivariateSpline
import json

from MECH import MECH
from mixture_properties import mixture_properties_with_temp


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


def rich_zone(
        power_arr: np.ndarray
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    
    with open(r'C:\Users\zksie\OneDrive - Delft University of Technology\Documents\Academic Year 2024-2025\Q4 - DSE\Emissions_tool\DSE_1\RQL_1D\rich_cases.json', 'r') as f:
        data = json.load(f)
        
    cases = data["cases"]
    power_plotted = np.array([case['power_kW'] for case in cases])
    NOx_rich_ppm = np.array([case['NOx_ppm'] for case in cases])
    P_const = np.array([case['P_inlet_Pa'] for case in cases])
    T_rich_exh = np.array([case['T_peak_K'] for case in cases])
    mdot_tot_points = np.array([case['TOTAL_AIR'] for case in cases])
    mdot_rich_in = np.array([case['mdot_air1'] for case in cases])
    mdot_rich_exh = np.array([case['mdot_exh'] for case in cases])
    

    NOx_spline = InterpolatedUnivariateSpline(power_plotted, NOx_rich_ppm, k=3)
    P_spline = InterpolatedUnivariateSpline(power_plotted, P_const, k=3)
    T_spline = InterpolatedUnivariateSpline(power_plotted, T_rich_exh, k=3)
    mdot_tot_spline = InterpolatedUnivariateSpline(power_plotted, mdot_tot_points, k=3)
    mdot_in_spline = InterpolatedUnivariateSpline(power_plotted, mdot_rich_in, k=3)
    mdot_exh_spline = InterpolatedUnivariateSpline(power_plotted, mdot_rich_exh, k=3)
    
    NOx_out =  NOx_spline(power_arr)
    P_out = P_spline(power_arr)
    T_out = T_spline(power_arr)
    mdot_tot = mdot_tot_spline(power_arr)
    mdot_in = mdot_in_spline(power_arr)
    mdot_exh = mdot_exh_spline(power_arr)

    quench_inputs = dict(P_out=P_out, T_out=T_out, mdot_tot_out=mdot_tot, mdot_in=mdot_in, mdot_exh=mdot_exh)

    return NOx_out, quench_inputs


def quench_zone(
        quench_inputs: Dict,
        power_arr: np.ndarray,
        mdot_h2o: np.ndarray,
        water_liquid: bool = True
        ) -> Tuple[List[float], np.ndarray]:
    
    NOx_normalized = np.full((len(power_arr), len(mdot_h2o)), np.nan)
    P_matrix = np.zeros((len(power_arr), len(mdot_h2o)))
    T_out = np.zeros((len(power_arr), len(mdot_h2o)))
    X_exh = np.empty((len(power_arr), len(mdot_h2o)), dtype=object)
    mdot_exh = np.zeros((len(power_arr), len(mdot_h2o)))
    mdot_air_2 = np.zeros(len(power_arr))

    with open(r'C:\Users\zksie\OneDrive - Delft University of Technology\Documents\Academic Year 2024-2025\Q4 - DSE\Emissions_tool\DSE_1\RQL_1D\rich_cases.json', 'r') as f:
        data = json.load(f)
    
    cases = data["cases"]
    power_plotted = np.array([case['power_kW'] for case in cases])
    X_in = np.array([case['X_exh'] for case in cases])
    
    T_in = quench_inputs['T_out']
    P_in = quench_inputs['P_out']
    mdot_in = quench_inputs['mdot_exh']

    for i in tqdm(range(len(power_arr))):
        
        idx_closest_power = np.argmin(np.abs(power_plotted - power_arr[i]))
        X_in_i = X_in[idx_closest_power]

        # Exhaust of the rich zone
        rich_exh = gas_obj(
                mdot=mdot_in[i],
                X=X_in_i,
                T=T_in[i],
                P=P_in[i]
                )

        # Air bled, to be mixed with rich exhaust to create stoichiometric mixture
        air = gas_obj(
                mdot=1.0,
                X="N2:0.78084, O2:0.20946",
                T=603.0,
                P=rich_exh.P
                )

        mdot_h2 = rich_exh.mdot * rich_exh.gas.X[1]
        mdot_o2 = 0.5 * mdot_h2  # stoichiometric ratio for H2 combustion, assuming no O2 in exhaust
        air.mdot = mdot_o2 / air.gas.X[3]
        mdot_air_2[i] = air.mdot

        mdot_mix1, T_mix1, P_mix1, X_mix1 = mixture_properties_with_temp(
            stream1=rich_exh,
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
                    P=rich_exh.P
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
            
            reactor = ct.IdealGasConstPressureReactor(stoich_with_water.gas)
            sim = ct.ReactorNet([reactor])

            # Time estimation based on assumed quench zone geometry
            V = stoich_with_water.mdot / (2*np.pi*0.15*0.02 * stoich_with_water.gas.density)
            tau_max = 0.1/V

            for t in np.arange(1e-9, tau_max, 1e-6):
                try:
                    sim.advance(t)

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

            P_matrix[i, j] = reactor.thermo.P
            T_out[i, j] = reactor.thermo.T
            X_exh[i, j] = reactor.thermo.X
            mdot_exh[i, j] = stoich_with_water.mdot

            O2_dry = reactor.thermo['O2'].X[0] / (1.0 - reactor.thermo['H2O'].X[0])
            F15 = (20.9 - 15.0) / (20.9 - 100.0 * O2_dry)
            
            NOx_normalized[i, j] = reactor.thermo['NO'].X[0] * F15 * 1e6

    lean_inputs = dict(mdot_exh=mdot_exh, X_exh=X_exh, T_out=T_out, P_out=P_matrix)

    return NOx_normalized, lean_inputs


def lean_zone(lean_inputs: Dict,
              power_arr: np.ndarray,
              mdot_h2o: np.ndarray,
              equivalence_ratio: float = 0.5):
    
    NOx_lean = np.zeros((len(power_arr), len(mdot_h2o)))
    mdot_air_3 = np.zeros((len(power_arr), len(mdot_h2o)))

    for i in range(len(power_arr)):
        for j in range(len(mdot_h2o)):

            lean_in = gas_obj(
                mdot=lean_inputs['mdot_exh'][i, j],
                X=lean_inputs['X_exh'][i, j],
                T=lean_inputs['T_out'][i, j],
                P=lean_inputs['P_out'][i, j]
            )
            print(lean_in.X)
            print(f"Equivalence ratio in lean inlet: {lean_in.gas.X[1] / (2 * lean_in.gas.X[3])}")

            air = gas_obj(
                    mdot=1.0,
                    X="N2:0.78084, O2:0.20946",
                    T=603.0,
                    P=lean_in.P
                    )

            mdot_h2 = lean_in.mdot * lean_in.gas.X[1]
            mdot_o2 = 0.5 * mdot_h2 / equivalence_ratio - lean_in.gas.X[3]
            mdot_o2 = mdot_o2 if mdot_o2 > 0 else 0.0
            air.mdot = mdot_o2 / air.gas.X[3]
            mdot_air_3[i, j] = air.mdot

            mdot_lean, T_lean, P_lean, X_lean = mixture_properties_with_temp(
                stream1=lean_in,
                stream2=air,
                MECH=MECH
            )

            lean_mix = gas_obj(
                mdot=mdot_lean,
                T=T_lean,
                P=P_lean,
                X=X_lean
            )

            print(f"Equivalence ratio in the lean zone: {lean_mix.gas.X[1] / (2 * lean_mix.gas.X[3])}, for air mass flow: {air.mdot}")

            lean_mix.gas.transport_model = "multicomponent"
            WIDTH_LEAN = 0.04

            flame_lean = ct.FreeFlame(lean_mix.gas, width=WIDTH_LEAN)
            flame_lean.set_refine_criteria(ratio=6.0, slope=0.01, curve=0.02)
            flame_lean.solve(loglevel=0, auto=True, refine_grid=True)

            O2_dry = flame_lean.X[0, 3] / (1.0 - flame_lean.X[0, 5])
            F15 = (20.9 - 15.0) / (20.9 - 100.0 * O2_dry)
            
            NOx_lean[i, j] = flame_lean.X[0, 9] * F15 * 1e6
            print(f'NOx [ppm]: ', NOx_lean[i, j])

    return NOx_lean, mdot_air_3


def main(
        power_arr: np.ndarray,
        mdot_h2o: np.ndarray,
        ) -> np.ndarray:
    print('RQL initiated.')
    NOx_rich, quench_inputs = rich_zone(power_arr)
    NOx_rich_matrix = np.tile(NOx_rich[:, None], (1, len(mdot_h2o)))
    print('Rich zone calculation completed...')
    NOx_quench, lean_inputs = quench_zone(quench_inputs=quench_inputs, power_arr=power_arr, mdot_h2o=mdot_h2o, water_liquid=True)
    print('Quick mix zone calculation completed...')
    # NOx_lean, _ = lean_zone(lean_inputs=lean_inputs, power_arr=power_arr, mdot_h2o=mdot_h2o)
    # print('Lean zone calculation completed...')

    NOx_tot = NOx_rich_matrix # + NOx_quench # + NOx_lean
    print('NOx calcualtion completed -> Plotting...')

    return NOx_tot



if __name__ == "__main__":
    power_arr = np.arange(64.5, 1223.0, 10)
    mdot_h2o = np.arange(0, 0.1 + 1e-3, 1e-3)

    NOx_tot = main(power_arr=power_arr, mdot_h2o=mdot_h2o)

    P_grid, W_grid = np.meshgrid(power_arr, mdot_h2o, indexing='ij')

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W_grid, P_grid, NOx_tot, cmap='viridis')

    # NOx requirement contour
    contour = ax.contour(
        W_grid, P_grid, NOx_tot,
        levels=[25],
        colors='red',
        linewidths=2,
        offset=25
    )
    ax.clabel(contour, fmt={25: '25 ppm'}, colors='red')

    ax.set_xlabel('Water injection [kg/s]')
    ax.set_ylabel('Power [kW]')
    ax.set_zlabel('NOx output [ppm]')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()


    # NOx_interpolator = RegularGridInterpolator(
    #     (T_rich_ex, mdot_h2o), NOx_normalized, bounds_error=False, fill_value=None
    # )

    # # os.makedirs('data/interpolants', exist_ok=True)

    # # Pickleeeeeeeeeeeee the interpolator
    # with open('data/interpolants/NOx_interpolator.pkl', 'wb') as f:
    #     pickle.dump(NOx_interpolator, f, protocol=pickle.HIGHEST_PROTOCOL)

