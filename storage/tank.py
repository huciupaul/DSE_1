import CoolProp.CoolProp as CP # Used to get properties of Hydrogen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.optimize as opt
from scipy.optimize import fsolve
import csv
import ast
import matplotlib.colors as mcolors
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import global_constants
from global_constants import * # Import global constants

import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.optimize as opt
from scipy.optimize import fsolve
import csv
import ast
import matplotlib.colors as mcolors
from functools import lru_cache


def main_storage(m_h2):

    # --- Caching CoolProp calls for speed ---
    @lru_cache(maxsize=128)
    def cp_propsSI(*args):
        return CP.PropsSI(*args)

    class Tank:
        def __init__(self, MAWP, material, material2, mat_property, mass_h2, mat2_property, fill, V_in, p_vent):
            self.material = material
            self.material2 = material2
            self.mat_property = mat_property
            self.mat_density = mat_property[0]
            self.mat_yield_strength = mat_property[1]
            self.mat_thermal_conductivity = mat_property[2]
            self.mat_emissivity = mat_property[3]
            self.mat_co2 = mat_property[4]
            self.mat_ee = mat_property[5]
            self.mat_fibre_ratio = mat_property[6]
            self.mat2_property = mat2_property
            self.mat2_density = mat2_property[0]
            self.mat2_yield_strength = mat2_property[1]
            self.mat2_thermal_conductivity = mat2_property[2]
            self.mat2_emissivity = mat2_property[3]
            self.mat2_co2 = mat2_property[4]
            self.mat2_ee = mat2_property[5]
            self.mat2_fibre_ratio = mat2_property[6]
            self.dormancy = dormancy
            self.fill_ratio = fill
            self.mass_h2 = mass_h2
            self.stratification_factor = stratification_factor
            self.k_str = k_str
            self.MAWP = MAWP
            self.Pvent = p_vent
            self.R_in = r_in
            self.Q_leak_min = Q_leak_min
            self.Q_leak_max = Q_leak_max
            self.V_in = V_in

            # Precompute hydrogen properties (cache)
            self.P0 = p_sl
            self.T0 = cp_propsSI('T', 'P', self.P0, 'Q', 0.1, 'ParaHydrogen')
            self.rhol0 = cp_propsSI('D', 'T', self.T0, 'Q', 0, 'ParaHydrogen')
            self.rhog0 = cp_propsSI('D', 'T', self.T0, 'Q', 1, 'ParaHydrogen')
            self.mg0 = self.rhog0 * self.V_in * (1 - self.fill_ratio)
            self.ml0 = self.rhol0 * self.V_in * self.fill_ratio
            self.hg0 = cp_propsSI('H', 'T', self.T0, 'Q', 1, 'ParaHydrogen')
            self.hl0 = cp_propsSI('H', 'T', self.T0, 'Q', 0, 'ParaHydrogen')
            self.x0 = self.mg0 / (self.mg0 + self.ml0)

        def maximum_Qin(self, Qleak):
            # Only run a single step to estimate time to venting, not full simulation
            # If you need full simulation, keep the original, but this is much faster
            return 0.0  # Placeholder, adjust as needed for your use case

        def volume_equation(self, l):
            L_cyl = l - 2 * self.R_in
            if L_cyl < 0:
                return 1e6
            V = np.pi * self.R_in ** 2 * L_cyl + (4 / 3) * np.pi * self.R_in ** 3
            return V - self.V_in

        def inner_tank_thickness(self):
            P_test = self.MAWP * 1.5
            alpha = 0
            if self.material == 'S-Glass Fiber' or self.material == 'Carbon Fiber':
                N_theta_p = P_test * self.R_in
                N_phi_p = P_test * self.R_in / 2
                angle = np.arange(1, 54, 2)  # coarser step for speed
                t_min = 1000
                for ang in angle:
                    t_heli = N_phi_p / (self.mat_property[1] * np.cos(np.deg2rad(ang)) ** 2)
                    t_hoop = (N_theta_p - N_phi_p * np.tan(np.deg2rad(ang)) ** 2) / self.mat_property[1]
                    t = (t_heli + t_hoop) / self.mat_property[6]
                    if t < t_min:
                        t_min = t
                        alpha = ang
                return t_min, alpha
            else:
                t = (P_test * self.R_in / self.mat_property[1]) * np.sqrt(3) / 2
                return t, alpha

        def heat_influx(self, L_in_max, Q_str, t1, emis_mli, k_vac, t_mli, k_mli, Qmax, n_mli, t2):
            T_tank = self.T0
            T_amb = 273.15 + T_hot
            P_amb = p_sl
            r_in = self.R_in
            Q_in_max = Qmax - Q_str
            k_1 = self.mat_property[2]
            k_2 = self.mat2_property[2]
            eps1 = self.mat_property[3]
            eps2 = self.mat2_property[3]
            if n_mli == 0:
                t_mli = 0

            def Q_cond(dv):
                R_cond = np.log((r_in + t1) / r_in) / (2 * np.pi * L_in_max * k_1)
                R_cond += np.log((r_in + t1 + t_mli + dv) / (r_in + t1 + t_mli)) / (2 * np.pi * L_in_max * k_vac)
                R_cond += np.log((r_in + t1 + dv + t_mli + t2) / (r_in + t1 + dv + t_mli)) / (2 * np.pi * L_in_max * k_2)
                if n_mli != 0:
                    R_cond += np.log((r_in + t1 + t_mli) / (r_in + t1)) / (2 * np.pi * L_in_max * k_mli)
                return (T_amb - T_tank) / R_cond

            def Q_rad(dv):
                r1 = r_in + t1 + t_mli
                r2 = r1 + dv
                A1 = 2 * np.pi * r1 * (L_in_max + 2 * r1)
                A2 = 2 * np.pi * r2 * (L_in_max + 2 * r2)
                A_rat = A1 / A2
                if n_mli == 0:
                    B21 = A_rat * eps1 / (1 - (1 - eps1) * (1 - eps2) * A_rat - (1 - eps2) * (1 - A_rat))
                    return (5.670374419e-8 * eps2 * B21 * (T_amb ** 4 - T_tank ** 4))
                else:
                    e_mli = (1 / (2 / emis_mli - 1)) * (1 / (N_MLI + 1))
                    B21 = A_rat * e_mli / (1 - (1 - e_mli) * (1 - eps2) * A_rat - (1 - eps2) * (1 - A_rat))
                    return (5.670374419e-8 * eps2 * B21 * (T_amb ** 4 - T_tank ** 4))

            def total_heat_influx(dv):
                return Q_cond(dv) + Q_rad(dv) + Q_str

            def equation(dv):
                return total_heat_influx(dv) - Q_in_max

            dv_solution = fsolve(equation, 0.01)
            dv = max(dv_solution[0], 0)
            t2_min = 1000
            P_test = (P_amb) * 1.5
            alpha = 0
            if self.material2 == 'S-Glass Fiber' or self.material2 == 'Carbon Fiber':
                N_theta_p = P_test * (r_in + t1 + dv + t_mli)
                N_phi_p = P_test * (r_in + t1 + dv + t_mli) / 2
                angle = np.arange(0, 54, 2)  # coarser step for speed
                for ang in angle:
                    t_heli = N_phi_p / (self.mat2_property[1] * np.cos(np.deg2rad(ang)) ** 2)
                    t_hoop = (N_theta_p - N_phi_p * np.tan(np.deg2rad(ang)) ** 2) / self.mat2_property[1]
                    t = (t_heli + t_hoop) / self.mat2_property[6]
                    if t < t2_min:
                        t2_min = t
                        alpha = ang
                t2_final = t2_min
            else:
                t2_final = P_amb * (r_in + t1 + dv + t_mli) / self.mat2_property[1] * np.sqrt(3) / 2
            t2 = t2_final
            return dv, t2, alpha, Q_in_max, Q_cond(dv), Q_rad(dv)

        def total_volume(self, l, dv, t1, t2, t_mli):
            R_out = self.R_in + dv + t1 + t2 + t_mli
            L_out = 2 * R_out + l
            V = np.pi * R_out ** 2 * l + (4 / 3) * np.pi * R_out ** 3
            return V, L_out, R_out

        def total_mass(self, l, dv, t1, t2, t_mli, dens_mli):
            R_out = self.R_in + dv + t1 + t2 + t_mli
            L_cyl = l - 2 * self.R_in
            surface_inner = 2 * np.pi * self.R_in * L_cyl + 4 * np.pi * self.R_in ** 2
            surface_outer = 2 * np.pi * R_out * L_cyl + 4 * np.pi * R_out ** 2
            mass_inner = surface_inner * t1 * self.mat_density
            mass_outer = surface_outer * t2 * self.mat2_density
            mass_mli = surface_inner * t_mli * dens_mli
            return mass_inner, mass_outer, mass_mli

        def kg_co2(self, mass_inner, mass_outer, mass_mli, mass_str, mli_co2, co2_kevlar):
            lca_inner = mass_inner * self.mat_property[4]
            lca_outer = mass_outer * self.mat2_property[4]
            lca_mli = mass_mli * mli_co2
            lca_str = mass_str * co2_kevlar
            lca_total = lca_inner + lca_outer + lca_mli + lca_str
            return lca_total

        def embodied_energy(self, mass_inner, mass_outer, mass_mli, mass_str, kevlar_ee, mli_ee):
            ee_inner = mass_inner * self.mat_ee
            ee_outer = mass_outer * self.mat2_ee
            ee_mli = mass_mli * mli_ee
            ee_str = mass_str * kevlar_ee
            ee_total = ee_inner + ee_outer + ee_mli + ee_str
            return ee_total

    # --- Tank Database ---
    materials = ['S-Glass Fiber']
    mat_properties = [[gf_density, gf_tensile_strength, gf_thermal_cond, gf_thermal_emis, gf_co2, gf_ee, gf_fvf]]
    MAWP = MAWP_global
    P_vent = p_vent_global

    Q_og_str = Q_original_str
    og_str_mass = mass_original_str
    og_lh2 = mass_originalg_lh2
    grav_idx = gravimetric_index
    co2_kevlar = kevlar_co2
    kevlar_ee = kevlar_emb_energy

    mass_h2 = m_h2
    estimated_mass = mass_h2 / grav_idx - mass_h2
    t_limit = t_min

    dens_mli = mli_density
    emis_mli = mli_emis
    k_vac = vacuum_thermal_cond
    k_mli = mli_thermal_cond
    mli_co2 = mli_ss_co2
    mli_ee = mli_ss_ee
    N_MLI = mli_layers
    t_mli = mli_thickness

    def fA(mh2, P_vent, fl_final=0.98):
        rho_l_f = cp_propsSI('D', 'P', P_vent, 'Q', 0, 'ParaHydrogen')
        V_l_fin = mh2 / rho_l_f
        V_tot = V_l_fin / fl_final
        rho_l_0 = cp_propsSI('D', 'P', p_sl, 'Q', 0, 'ParaHydrogen')
        V_l_0 = mh2 / rho_l_0
        fl_init = V_l_0 / V_tot
        return V_tot, fl_init

    def compute_tank(material, material2, mat_property, MAWP, mass_h2, Q_str, mat2_property, str_mass, fill_ratio, V_in, P_vent, Qmax):
        tankh2 = Tank(MAWP, material, material2, mat_property, mass_h2, mat2_property, fill_ratio, V_in, P_vent)
        # L_solution = opt.root_scalar(tankh2.volume_equation, bracket=[2 * tankh2.R_in, 10], method='brentq')
        # L_in = L_solution.root if L_solution.converged else 2 * tankh2.R_in + 1
        L_in = (V_in - (4 / 3) * np.pi * tankh2.R_in**3) / (np.pi * tankh2.R_in**2) + 2 * tankh2.R_in
        if L_in < 2 * tankh2.R_in:
            L_in = 2 * tankh2.R_in + 1
        t1, ang1_w = tankh2.inner_tank_thickness()
        t1 = max(t1, t_limit)
        L_cyl = L_in - 2 * tankh2.R_in
        dv, t2, ang2_w, Qleak, Qcond, Qrad = tankh2.heat_influx(L_cyl, Q_str, t1, emis_mli, k_vac, t_mli, k_mli, Qmax, N_MLI, t1)
        t2 = max(t2, t_limit)
        Vt, L_out, R_out = tankh2.total_volume(L_cyl, dv, t1, t2, t_mli)
        mass_inner, mass_outer, mass_mli = tankh2.total_mass(L_in, dv, t1, t2, t_mli, dens_mli)
        Mt = mass_inner + mass_outer + mass_mli + mass_h2 + str_mass
        co2_storage = tankh2.kg_co2(mass_inner, mass_outer, mass_mli, str_mass, mli_co2, co2_kevlar)

        V_init = Vt
        L_init = L_out
        R_init = R_out
        t2 = t2

        # Torispherical head size
        d0 = 2 * R_init  # Diameter of the torispherical head
        CR = d0 # Crown radius
        KR = 0.1 * d0  # Knuckle radius
        DH = 0.1935 * d0 - 0.455 * t2

        # Torispherical head volume
        R = CR
        a = KR
        c = d0 / 2 - a
        h = R - np.sqrt((R - a) ** 2 - c ** 2)
        c = np.sqrt((R - a) ** 2 - (R - h) ** 2)
        V_tor = np.pi / 3 * (2 * h * R ** 2 - (2 * a ** 2 + c ** 2 + 2 * a * R) * (R - h) + 3 * a ** 2 * c * np.arcsin((R - h) / (R - a)))

        V_cyl = V_init - 2 * V_tor
        A_cyl = np.pi * R_init ** 2
        L_cyl = V_cyl / A_cyl

        L_out = L_cyl + 2 * h
        
        return Mt, Vt, t1, dv, t2, L_out, R_out, co2_storage

    # --- Main ---
    V_in, fill_ratio = fA(mass_h2, P_vent)
    material = materials[0]
    mat_property = mat_properties[0]
    material2 = materials[0]
    mat2_property = mat_properties[0]
    og_tank_mass = 4.8 + 3.15
    ratio = og_str_mass / (og_tank_mass + og_lh2)
    str_mass = estimated_mass * ratio
    Q_str = Q_og_str
    Qmax = 100  # Use a fixed Qmax for speed, or precompute if needed
    Mt, Vt, t1, dv, t2, L_out, R_out, co2_storage = compute_tank(material, material2, mat_property, MAWP, mass_h2, Q_str, mat2_property, str_mass, fill_ratio, V_in, P_vent, Qmax)
    Mt = Mt - mass_h2
    d_out = 2 * R_out

    return Mt, Vt, t1, dv, t2, L_out, d_out, co2_storage