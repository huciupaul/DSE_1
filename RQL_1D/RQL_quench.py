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
from scipy.interpolate import RegularGridInterpolator


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
        T_rich_ex: np.ndarray,
        mdot_h2o: np.ndarray
        ) -> Tuple[List[float], np.ndarray]:

    mdot_NOx = np.zeros((len(T_rich_ex), len(mdot_h2o)))

    # temps_before_reaction = np.zeros((len(T_rich_ex), len(mdot_h2o)))
    # temps_after_reaction = np.zeros((len(T_rich_ex), len(mdot_h2o)))
    # times_to_steady = np.zeros((len(T_rich_ex), len(mdot_h2o)))

    for i, T_rich_ex_i in tqdm(enumerate(T_rich_ex)):
        # Exhaust of the rich zone
        rich_ex = gas_obj(
                mdot=0.23071218802678672,
                X="N2:0.27396853378378067, H2:0.5784925158354742, H:0.0005133907896591396, O2:7.393652373124469e-10, O:4.591482831748451e-09, H2O:0.14701811528905126, OH:7.430992453322465e-06, H2O2:1.1742925935728515e-10, HO2:3.6285427485688574e-12, NO:5.4364866743951985e-09, N2O:9.901157119223157e-12, N:4.953608429224339e-12, NH:1.4251848100599631e-11, NNH:1.977256219409285e-09, NH2:4.148258601540983e-10",
                T=T_rich_ex_i,
                P=12 * ct.one_atm
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
            lh2o = gas_obj(
                    mdot=m_dot_h2o_j,
                    X="H2O:1",
                    T=353.0,
                    P=rich_ex.P
                    )
            
            mdot_mix, T_mix, P_mix, X_mix = mixture_properties_with_temp(
                stream1=stoich_mix,
                stream2=lh2o,
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
            # temps_before_reaction[i, j] = reactor.thermo.T

            try:
                sim.advance_to_steady_state()
                # times_to_steady[i, j] = sim.time

            except Exception as e:
                print("Error at m_dot_h2o_i =", m_dot_h2o_j)
                print("T =", stoich_with_water.gas.T)
                print("P =", stoich_with_water.gas.P)
                print("X =", stoich_with_water.gas.X)
            
                if reactor.thermo.T > 20000 or reactor.thermo.T < 200:
                    print(f"Unphysical T = {reactor.thermo.T}, skipping this iteration.")
                    print(f"Reactor state before failure:\nT = {reactor.thermo.T}, Y = {reactor.thermo.Y}")
                    print(f"Sum(Y) = {sum(reactor.thermo.Y)}")

                    
            # temps_after_reaction[i, j] = reactor.thermo.T
            mdot_NOx[i, j] = reactor.thermo['NO'].X[0] * stoich_with_water.mdot # kg/s

    # return mdot_NOx, temps_before_reaction, temps_after_reaction, times_to_steady, T_rich_ex, mdot_h2o, mdots
    return mdot_NOx


if __name__ == "__main__":
    T_rich_ex = np.arange(1570, 1740, 1)
    mdot_h2o = np.arange(0, 0.1 + 1e-3, 1e-2)

    mdot_NOx = main(T_rich_ex=T_rich_ex, mdot_h2o=mdot_h2o)

    T_grid, W_grid = np.meshgrid(T_rich_ex, mdot_h2o, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W_grid, T_grid, mdot_NOx, cmap='viridis')

    ax.set_xlabel('Water injection [kg/s]')
    ax.set_ylabel('Rich zone exhaust temperature [K]')
    ax.set_zlabel('NOx formed [kg/s]')
    ax.set_title('NOx vs Water Injection and Rich Zone Temperature')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

    NOx_interpolator = RegularGridInterpolator(
        (T_rich_ex, mdot_h2o), mdot_NOx, bounds_error=False, fill_value=None
    )

    os.makedirs('data/interpolants', exist_ok=True)

    # Pickleeeeeeeeeeeee the interpolator
    with open('data/interpolants/NOx_interpolator.pkl', 'wb') as f:
        pickle.dump(NOx_interpolator, f, protocol=pickle.HIGHEST_PROTOCOL)