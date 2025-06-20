# import sys
# sys.path.append("..")  # Add parent directory to path

import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fpp.turboprop import Turboprop
import global_constants
from global_constants import MAXC, TOGA, base_AP, G_0, E_SCALE, R_AIR, k_air, isa_atmosphere
from fc.fc_for_wrapper import FuelCell
# Dataclass definitions
@dataclass
class Aircraft:
    name: str
    wing_area: float
    wing_span: float
    CD0: float
    prop_diameter: float
    eng: Turboprop
    fc: FuelCell
    MTOW: float
    D_RAD: float = 0.0  # [N] Drag penalty due to radiator 
    P_cc_min: float = 0.0
    MAXC: float = MAXC
    TOGA: float = TOGA # [W] Take-Off/Go-Around propulsive power
    base_AP: float = base_AP  # Operational auxiliary power
    delta_AP: float = 0.0  # Thermal Management System power
    n_prop: int = 2  # Added n_prop for number of propellers
    
    @property
    def aspect_ratio(self) -> float:
        return self.wing_span ** 2 / self.wing_area

    @property
    def oswald(self) -> float:
        # From Mohammad Sadraey, "Wing Design"
        return 1.78 * (1 - 0.045 * self.aspect_ratio ** 0.68) - 0.64

    @property
    def prop_area(self) -> float:
        return self.n_prop * np.pi * (self.prop_diameter * 0.5) ** 2
    
@dataclass
class Waypoint:
    name: str
    altitude: float   # [m] Altitude at start of phase
    airspeed: float   # [m/s] Airspeed at start of phase
    roc: float        # [m/s] ROC/D during phase
    hold_time: float = 0.0  # [s]
    nominal: bool = True  # True if this is a nominal waypoint

@dataclass
class Powerpoint:
    name: str
    power: float=None # [W]
    time: float=None  # [s]
    until_phase: str=None   # e.g. "cruise"
    
class FlightMission:
    """
    Time-resolved kinematic + power mission profile.

    If *total_range_m* is provided the cruise leg is auto-sized to achieve the
    specified ground range. Cruise and hold power settings are left `None` and
    filled via the placeholder `placeholder_power_flat_section()` to allow the
    user to slot in their own algorithm later.
    """

    def __init__(self,
                 aircraft: Aircraft,
                 waypoints: List[Waypoint],
                 powerpoints: List[Powerpoint],
                 R_LHV: float = 1.0,
                 dt: float = 1.0,
                 total_range_m: Optional[float] = None,
                 cruise_wp_name: str = "cruise",
                 throttle_cruise: float = 0.1) -> None:
        if len(waypoints) < 2:
            raise ValueError("Need at least two waypoints.")
        self.ac = aircraft
        self.wps = waypoints
        self.pws = powerpoints
        self.R_LHV = R_LHV
        self.dt = dt
        self.total_range = total_range_m
        self.cruise_wp_name = cruise_wp_name.lower()
        self.throttle_cruise = throttle_cruise  # Cruise fuel cell throttle setting
        self._profile: Dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------------
    #  PUBLIC INTERFACE
    # ---------------------------------------------------------------------
    @property
    def profile(self) -> Dict[str, np.ndarray]:
        if not self._profile:
            self.__build()
        return self._profile

    # ---------------------------------------------------------------------
    #  INTERNAL HELPERS – kinematics
    # ---------------------------------------------------------------------
    @staticmethod
    def __ground_speed(V: float, roc: float) -> float:
        if V == 0:
            return 0.0
        gamma = np.arcsin(np.clip(roc / V, -1.0, 1.0))
        return V * np.cos(gamma)
    
    def __drag_balance(self, m: float, rho_inf: float, V_inf:float):
        S = self.ac.wing_area
        AR = self.ac.aspect_ratio
        C_D0 = self.ac.CD0
        e = self.ac.oswald
        
        # Constant altitude: Lift = Weight
        C_L = m*G_0/(0.5*rho_inf*V_inf**2*S)
        
        # Induced drag coefficient
        C_Di = C_L**2/(np.pi*e*AR)
        
        # Total drag coefficient
        C_D = C_D0 + C_Di
                
        # Constant velocity: Thrust = Drag
        T = 0.5*rho_inf*V_inf**2*S*C_D
        P = T*V_inf
        
        return P
    
    # ---------------------------------------------------------------------
    #  INTERNAL HELPERS – fuel cell
    # ---------------------------------------------------------------------
    
    @staticmethod
    def __mdot_fc(Pa: float):
        """
        Compute mass flow rate of fuel from the fuel cell available power.
        Args:
            Pa: Available power [W]
        Returns:
            mdot_fc: Mass flow rate of fuel [kg/s]
        """
        return 0.01560072557 * Pa/1e6  # 0.01560072557 kg/s per MW
     
    # ---------------------------------------------------------------------
    #  CORE MISSION BUILDER
    # ---------------------------------------------------------------------
    def __pp_slicer(self,
               phase_arr: np.ndarray,
               time_arr: np.ndarray) -> List[Tuple[Powerpoint, slice]]:
        """Return a list of (Powerpoint, slice) covering the whole mission."""
        pp_slices = []
        i0 = 0

        for k, pp in enumerate(self.pws):
            # --- determine i1 (index AFTER the segment) ---
            if pp.time is not None:                       # duration‑based
                t_end = time_arr[i0] + pp.time
                i1 = np.searchsorted(time_arr, t_end, side="right")

            elif pp.until_phase is not None:              # phase‑based
                mask = phase_arr[i0:] == pp.until_phase.lower()
                if not mask.any():
                    raise ValueError(f"Phase ‘{pp.until_phase}’ never reached.")
                i1 = i0 + np.argmax(mask)                 # first element == True

            else:                                         # default: until next pp or end
                if k < len(self.pws) - 1:
                    # wait for next pp’s phase/time to start
                    continue
                i1 = len(time_arr)

            pp_slices.append((pp, slice(i0, i1)))
            i0 = i1

        # catch the fall‑through when we deferred the “until next pp” case
        if len(pp_slices) < len(self.pws):
            pp_prev, sl_prev = pp_slices[-1]
            pp_slices[-1] = (pp_prev, slice(sl_prev.start, len(time_arr)))

        if pp_slices[-1][1].stop != len(time_arr):
            raise RuntimeError("Powerpoints don’t cover entire mission.")
                               
        return pp_slices
    
    def __build(self) -> None:
        # Containers
        V_full, h_full, roc_full, time_full, phase_full = [], [], [], [], []
        t = 0.0
        dist_accum = 0.0  # [m]

        # Helper to append one step
        def __append(V_s: float, h_s: float, roc_s: float, phase_name: str):
            nonlocal t, dist_accum
            V_full.append(V_s)
            h_full.append(h_s)
            roc_full.append(roc_s)
            time_full.append(t)
            phase_full.append(phase_name)
            dist_accum += self.__ground_speed(V_s, roc_s) * self.dt
            t += self.dt

        # Identify cruise waypoint
        try:
            i_cruise = next(i for i, wp in enumerate(self.wps) if wp.name.lower() == self.cruise_wp_name)
        except StopIteration:
            if self.total_range is not None:
                raise ValueError("Cruise waypoint not found; needed for range sizing.")
            i_cruise = None

        # ------------------------------------------------------------------
        # 1) Build climb + any cruise hold + (possibly) partial descent
        # ------------------------------------------------------------------
        for idx, wp in enumerate(self.wps):
            # Insert hold at current waypoint (except idx == 0)
            if wp.hold_time > 0:
                n_hold = int(np.ceil(wp.hold_time / self.dt))
                for _ in range(n_hold):
                    __append(wp.airspeed, wp.altitude, 0.0, wp.name)

            # Last waypoint → nothing to transition to
            if idx == len(self.wps) - 1:
                break
            nxt = self.wps[idx + 1]

            # If we are at cruise wp and total_range sizing requested → break
            if idx == i_cruise and self.total_range is not None:
                break

            # Compute differential quantities
            dh = nxt.altitude - wp.altitude
            dv = nxt.airspeed - wp.airspeed
            roc_cmd = wp.roc if wp.roc != 0 else nxt.roc

            # ---- Constant‑altitude leg -----------------------------------
            if abs(dh) < 1e-6:
                # Taxi phases jump instantly to next speed
                if "taxi" in wp.name.lower() or "taxi" in nxt.name.lower():
                    __append(nxt.airspeed, wp.altitude, 0.0, wp.name)
                    continue
                
                # Otherwise interpolate speed change
                if abs(dv) < 1e-6:
                    continue
                n_steps = max(int(np.ceil(abs(dv) / 1.0)), 1)
                V_seg = np.linspace(wp.airspeed, nxt.airspeed, n_steps, endpoint=False)
                for V_s in V_seg:
                    __append(V_s, wp.altitude, 0.0, wp.name)
                continue

            # ---- Climb / descent ----------------------------------------
            if roc_cmd == 0:
                raise ValueError(f"No ROC defined for leg {wp.name} → {nxt.name}.")
            n_steps = int(np.ceil(abs(dh / roc_cmd) / self.dt))
            alt_seg = np.linspace(wp.altitude, nxt.altitude, n_steps, endpoint=False)
            V_seg = np.linspace(wp.airspeed, nxt.airspeed, n_steps, endpoint=False)
            for V_s, h_s in zip(V_seg, alt_seg):
                __append(V_s, h_s, roc_cmd, wp.name)

        # ------------------------------------------------------------------
        # 2) Auto‑size cruise if requested, then build remaining legs
        # ------------------------------------------------------------------
        if self.total_range is not None:
            V_post, h_post, roc_post, phase_post = [], [], [], []
            dist_post = 0.0
            wp_start = self.wps[i_cruise]

            for idx in range(i_cruise, len(self.wps) - 1):
                wp = self.wps[idx]
                nxt = self.wps[idx + 1]
                should_count_range = wp.nominal and nxt.nominal

                # Hold at wp (except cruise wp itself)
                if idx != i_cruise and wp.hold_time > 0:
                    n_hold = int(np.ceil(wp.hold_time / self.dt))
                    for _ in range(n_hold):
                        V_post.append(wp.airspeed)
                        h_post.append(wp.altitude)
                        roc_post.append(0.0)
                        phase_post.append(wp.name)
                        if wp.nominal:
                            dist_post += self.__ground_speed(wp.airspeed, 0.0) * self.dt

                # Transition to next
                dh = nxt.altitude - wp.altitude
                roc_cmd = wp.roc if wp.roc != 0 else nxt.roc
                if abs(dh) < 1e-6:
                    if "taxi" in wp.name.lower() or "taxi" in nxt.name.lower():
                        V_post.append(nxt.airspeed)
                        h_post.append(wp.altitude)
                        roc_post.append(0.0)
                        phase_post.append(wp.name)
                        if should_count_range:
                            dist_post += self.__ground_speed(nxt.airspeed, 0.0) * self.dt
                        continue
                    dv = nxt.airspeed - wp.airspeed
                    if abs(dv) < 1e-6:
                        continue
                    n_steps = max(int(np.ceil(abs(dv) / 1.0)), 1)
                    V_seg = np.linspace(wp.airspeed, nxt.airspeed, n_steps, endpoint=False)
                    for V_s in V_seg:
                        V_post.append(V_s)
                        h_post.append(wp.altitude)
                        roc_post.append(0.0)
                        phase_post.append(wp.name)
                        if should_count_range:
                            dist_post += self.__ground_speed(V_s, 0.0) * self.dt
                    continue

                if roc_cmd == 0:
                    raise ValueError(f"No ROC for leg {wp.name} → {nxt.name}.")
                n_steps = int(np.ceil(abs(dh / roc_cmd) / self.dt))
                alt_seg = np.linspace(wp.altitude, nxt.altitude, n_steps, endpoint=False)
                V_seg = np.linspace(wp.airspeed, nxt.airspeed, n_steps, endpoint=False)
                for V_s, h_s in zip(V_seg, alt_seg):
                    V_post.append(V_s)
                    h_post.append(h_s)
                    roc_post.append(roc_cmd)
                    phase_post.append(wp.name)
                    if should_count_range:
                        dist_post += self.__ground_speed(V_s, roc_cmd) * self.dt

            # Distance left for cruise
            dist_remaining = self.total_range - dist_accum - dist_post
            if dist_remaining < 0:
                raise ValueError(f"Mission exceeds total range by {-dist_remaining/1000:.1f} km.")
            if dist_remaining > 0:
                V_cruise = wp_start.airspeed
                n_cruise = int(np.ceil(dist_remaining / (V_cruise * self.dt)))
                for _ in range(n_cruise):
                    __append(V_cruise, wp_start.altitude, 0.0, wp_start.name)
            # Append post‑segments
            for V_s, h_s, roc_s, ph_s in zip(V_post, h_post, roc_post, phase_post):
                __append(V_s, h_s, roc_s, ph_s)

        # ------------------------------------------------------------------
        # 2b) Add any hold at the **final** waypoint (previously skipped)
        # ------------------------------------------------------------------
        wp_last = self.wps[-1]
        if wp_last.hold_time > 0:
            n_hold = int(np.ceil(wp_last.hold_time / self.dt))
            for _ in range(n_hold):
                __append(wp_last.airspeed, wp_last.altitude, 0.0, wp_last.name)
        # ------------------------------------------------------------------
        # 3) Force descent to ground if final waypoint altitude > 0
        # ------------------------------------------------------------------
        if h_full and h_full[-1] > 0.0:
            last_V = V_full[-1]
            roc_land = self.wps[-1].roc if self.wps[-1].roc != 0 else -3.0
            roc_land = -abs(roc_land)
            while h_full[-1] > 0.0:
                next_alt = h_full[-1] + roc_land * self.dt
                if next_alt <= 0.0:
                    next_alt = 0.0
                    __append(last_V, next_alt, 0.0, self.wps[-1].name)
                    break
                __append(last_V, next_alt, roc_land, self.wps[-1].name)

        # ------------------------------------------------------------------
        # 4) Export profile dictionary
        # ------------------------------------------------------------------
        time_arr = np.asarray(time_full)
        V_arr = np.asarray(V_full)
        h_arr = np.asarray(h_full)
        roc_arr = np.asarray(roc_full)
        phase_arr = np.asarray(phase_full)
        
        # ------------------------------------------------------------------
        # 5) Compute power profile
        # ------------------------------------------------------------------
        T_arr, P_arr, rho_arr, a_arr    = isa_atmosphere(h_arr)
        Pr_arr                          = np.zeros_like(V_arr)
        Pa_arr                          = np.zeros_like(V_arr)
        P_cc_arr                        = np.zeros_like(V_arr)
        P_fc_arr                        = np.zeros_like(V_arr)
        m_arr                           = np.zeros_like(V_arr); m_arr[0] = self.ac.MTOW
        mdot_fuel_arr                   = np.zeros_like(V_arr)
        mdot_air_arr                    = np.zeros_like(V_arr)
        mdot_fc_arr                     = np.zeros_like(V_arr)
        mdot_cc_arr                     = np.zeros_like(V_arr)
        mdot_dumpy_arr                  = np.zeros_like(V_arr)
        m_dumpy                         = np.zeros_like(V_arr)
        eta_th_arr                      = np.zeros_like(V_arr)
        eta_prop_arr                    = np.zeros_like(V_arr)
        T_cc                            = np.zeros_like(V_arr)
        p_cc                            = np.zeros_like(V_arr)
        Qdot_fc                         = np.zeros_like(V_arr)
        mdot_fc_air_in                  = np.zeros_like(V_arr)
        mdot_fc_air_out                 = np.zeros_like(V_arr)
        mdot_fc_H2O                     = np.zeros_like(V_arr)
        mdot_fc_H2_recirculation        = np.zeros_like(V_arr)
        mdot_NOx_arr                    = np.zeros_like(V_arr)
        m_NOx_arr                       = np.zeros_like(V_arr)
        

        # Create IDLE, TOGA, CRUISE generalized masks
        mask_idle   = (phase_arr == "taxi\\to") | (phase_arr == "taxi\\landing")
        mask_toga   = (phase_arr == "takeoff") | (phase_arr == "climb1") | (phase_arr == "climb2")
        mask_cruise = ~(mask_idle | mask_toga)
        
        throttle = mask_toga * self.ac.fc.throttle_TOGA + ~mask_toga * self.throttle_cruise
        
        for pp, sl in self.__pp_slicer(phase_arr, time_arr):
            if pp.power is not None:
                # Powerpoint with power
                Pa_total = pp.power * self.ac.MAXC
                P_fc_max = self.ac.fc.power_max_throttle * throttle[sl][0]

                if Pa_total <= P_fc_max:
                    Pa_fc = self.ac.fc.power_min if Pa_total < self.ac.fc.power_min else Pa_total
                    Pa_cc = 0.0
                    P_fc_dumpy = Pa_fc - Pa_total
                else:
                    Pa_cc = Pa_total - P_fc_max
                    Pa_fc = P_fc_max
                    P_fc_dumpy = 0.0
                    if Pa_cc < self.ac.P_cc_min:
                        Pa_cc = self.ac.P_cc_min
                        Pa_fc = Pa_total - Pa_cc
                        if Pa_fc < self.ac.fc.power_min:
                            Pa_fc = self.ac.fc.power_min
                            Pa_cc = self.ac.P_cc_min
                            P_fc_dumpy = (Pa_cc + Pa_fc) - Pa_total
                
                Pa_cc_arr = Pa_cc * np.ones_like(V_arr[sl])
                Pa_arr[sl] = Pa_total   

                P_cc_arr[sl] = Pa_cc
                P_fc_arr[sl] = Pa_fc

                # Add fuel cell mass flow from threshold power
                Qdot_fc[sl], mdot_fc_arr[sl], mdot_fc_air_in[sl], mdot_fc_air_out[sl], mdot_fc_H2O[sl], mdot_fc_H2_recirculation[sl], mdot_cc_H2O = self.ac.fc.get_TMS_values(power=Pa_fc)
                
                # Fuel mass flow using Torenbeek method (PSFC)
                mdot_fuel_arr[sl], eta_th_arr[sl], eta_prop_arr[sl], mdot_air_arr[sl], T_cc[sl], p_cc[sl], mdot_NOx_arr[sl] = self.ac.eng.compute(
                    T_arr[sl], P_arr[sl], rho_arr[sl], V_arr[sl], R_LHV=self.R_LHV, Pa=Pa_cc_arr, mdot_H2O=mdot_cc_H2O)
                mdot_cc_arr[sl] = mdot_fuel_arr[sl]
                
  
                mdot_fuel_arr[sl] += mdot_fc_arr[sl]

                mdot_dumpy_arr[sl] = mdot_fc_arr[sl] * P_fc_dumpy / Pa_fc
                m_dumpy[sl] = mdot_dumpy_arr[sl] * self.dt  # cumulative fuel dumped _within_ slice
                
                # Required power
                Pr_arr[sl] = Pa_arr[sl] * eta_prop_arr[sl]
                
                m_start = m_arr[sl.start - 1] if sl.start > 0 else self.ac.MTOW
                burn = np.cumsum(mdot_fuel_arr[sl]) * self.dt          # cumulative fuel burnt _within_ slice
                m_arr[sl] = m_start - np.concatenate(([0.0], burn[:-1]))
                
                m_NOx_start = m_NOx_arr[sl.start - 1] if sl.start > 0 else 0.0
                m_NOx_arr[sl] = m_NOx_start + np.cumsum(mdot_NOx_arr[sl]) * self.dt
                
        
            else:
                # Compute power from drag balance
                # --- initialise mass for this slice so it’s continuous ----------
                m_arr[sl.start] = m_arr[sl.start - 1] if sl.start > 0 else self.ac.MTOW
                # Order of operations:
                # 1) Extract weight @previous time step
                # 2) Compute thrust required @current time step
                # 3) Compute propulsive efficiency @current time step
                # 4) Compute available power @current time step
                # 5) Compute fuel flow and thermal efficiency @current time step
                # 6) Integrate fuel flow to get mass @current time step
                # 7) Store results @current time step
                # Repeat for next time step
                for i in range(sl.start, sl.stop):
                    m_curr = m_arr[i]
                    Pr = self.__drag_balance(m_curr, rho_arr[i], V_arr[i])
                    
                    # Hijack turboprop propeller efficiency internal method to compute available power
                    a0 = np.sqrt(k_air * R_AIR * T_arr[i])
                    M0 = V_arr[i] / a0
                    eta_prop = self.ac.eng._eta_prop(M0)
                    
                    Pa_total = Pr / eta_prop
                    P_fc_max = self.ac.fc.power_max_throttle * throttle[i]
                    
                    if Pa_total <= P_fc_max:
                        Pa_fc = self.ac.fc.power_min if Pa_total < self.ac.fc.power_min else Pa_total
                        Pa_cc = 0.0
                        P_fc_dumpy = Pa_fc - Pa_total
                    else:
                        Pa_cc = Pa_total - P_fc_max
                        Pa_fc = P_fc_max
                        P_fc_dumpy = 0.0
                        if Pa_cc < self.ac.P_cc_min:
                            Pa_cc = self.ac.P_cc_min
                            Pa_fc = Pa_total - Pa_cc
                            if Pa_fc < self.ac.fc.power_min:
                                Pa_fc = self.ac.fc.power_min
                                Pa_cc = self.ac.P_cc_min
                                P_fc_dumpy = (Pa_cc + Pa_fc) - Pa_total
                    # Add fuel cell mass flow from threshold power
                    Qdot_fc[i], mdot_fc_arr[i], mdot_fc_air_in[i], mdot_fc_air_out[i], mdot_fc_H2O[i], mdot_fc_H2_recirculation[i], mdot_cc_H2O = self.ac.fc.get_TMS_values(power=Pa_fc)
                    
                    mdot_fuel, eta_th, eta_prop, mdot_air, T_cc[i], p_cc[i], mdot_NOx_arr[i] = self.ac.eng.compute(
                        T_arr[i], P_arr[i], rho_arr[i], V_arr[i], R_LHV=self.R_LHV, Pa=Pa_cc, mdot_H2O=mdot_cc_H2O
                    )
                    mdot_cc_arr[i] = mdot_fuel
                    

                    mdot_fuel += mdot_fc_arr[i]

                    mdot_dumpy_arr[i] = mdot_fc_arr[i] * P_fc_dumpy / Pa_fc
                    m_dumpy[i+1] = m_dumpy[i] + mdot_dumpy_arr[i] * self.dt

                    m_arr[i+1] = m_arr[i] - mdot_fuel*self.dt
                    m_NOx_arr[i+1] = m_NOx_arr[i] + mdot_NOx_arr[i] * self.dt
                    
                    
                    # 7) Store results @current time step
                    Pr_arr[i] = Pr
                    Pa_arr[i] = Pa_total
                    P_cc_arr[i] = Pa_cc
                    P_fc_arr[i] = Pa_fc
                    mdot_fuel_arr[i] = mdot_fuel
                    mdot_air_arr[i] = mdot_air
                    eta_th_arr[i] = eta_th
                    eta_prop_arr[i] = eta_prop

        # finally
        self.TMS_inputs = dict(Q_dot_fc=Qdot_fc, p_cc=p_cc,
                           h2_mf_fc=mdot_fc_arr, h2_mf_cc=mdot_cc_arr,
                           t_cc=T_cc, air_mf_fc=mdot_fc_air_in,
                           t_amb=T_arr, rho_amb=rho_arr, V_amb=V_arr,
                           P_amb=P_arr,
                           h2o_mf_fc=mdot_fc_H2O, h2_mf_fc_recirculated=mdot_fc_H2_recirculation
                           )
        
        self._profile = dict(time=time_arr, V=V_arr, alt=h_arr,
                            T=T_arr, P=P_arr, rho=rho_arr, ROC=roc_arr,
                            phase=phase_arr, Pa=Pa_arr, Pr=Pr_arr,
                            P_cc=P_cc_arr, P_fc=P_fc_arr,
                            mdot_fuel=mdot_fuel_arr, mdot_air=mdot_air_arr,
                            mdot_cc=mdot_cc_arr, mdot_fc=mdot_fc_arr,
                            mdot_dumpy=mdot_dumpy_arr, m_dumpy=m_dumpy,
                            eta_th=eta_th_arr, eta_prop=eta_prop_arr, mass=m_arr,
                            mdot_cc_H2O=mdot_cc_H2O,
                            mdot_NOx=mdot_NOx_arr,
                            m_NOx=m_NOx_arr,
                            )

    # ---------------------------------------------------------------------
    #  QUICK‑LOOK PLOTS (kinematics + power)
    # ---------------------------------------------------------------------
    def quicklook(self) -> None:
        p = self.profile

        fig, axs = plt.subplots(3, 4, figsize=(18, 4))
        axs[0, 0].plot(p["time"], p["alt"])
        axs[0, 0].set_ylabel("Altitude [m]")
        axs[0, 1].plot(p["time"], p["V"])
        axs[0, 1].set_ylabel("TAS [m/s]")
        axs[0, 2].plot(p["time"], p["ROC"])
        axs[0, 2].set_ylabel("ROC [m/s]")
        axs[0, 3].plot(p["time"], p["Pa"])
        axs[0, 3].set_ylabel("Shaft power [W]")
        axs[1, 0].plot(p["time"], p["Pr"])
        axs[1, 0].set_ylabel("Power [W]")
        axs[1, 1].plot(p["time"], p["mdot_fuel"])
        axs[1, 1].set_ylabel("Fuel flow [kg/s]")
        axs[1, 2].plot(p["time"], p["eta_th"])
        axs[1, 2].set_ylabel("Thermal efficiency [-]")
        axs[1, 3].plot(p["time"], p["eta_prop"])
        axs[1, 3].set_ylabel("Propulsive efficiency [-]")
        axs[2, 0].plot(p["time"], p["mass"])
        axs[2, 0].set_ylabel("Mass [kg]")
        axs[2, 1].plot(p["time"], p["mdot_air"])
        axs[2, 1].set_ylabel("Air mass flow [kg/s]")
        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
        
    
def fpp_main(fc_split: float=0.0, throttle_TOGA: float = 0.85, throttle_cruise: float = 0.1, MTOW: float=8037.6, CD_RAD: float=0.0, delta_AP: float=0.0, dt: float=0.1) -> tuple:
    """
    Main flight performance function to obtain the fuel mass and shaft power profile.
    Args:
        
        MTOW: Maximuam Take-Off Weight in kg.
        CD_HEX: Coefficient of drag for the heat exchangers fuselage.
        
    Returns:
        tuple: A tuple containing the fuel mass required for the mission and the maximum fuel cell power.
    """
    wps = [
        Waypoint("taxi\\TO", 0.0, 8.231111, 0.0, hold_time=10 * 60),
        Waypoint("takeoff", 0.0, 54.0167, 797 / 60),
        Waypoint("climb1", 2_438.4, 61.73333, 797 / 60),
        Waypoint("climb2", 4_876.8, 61.73333, 797 / 60),
        Waypoint("cruise", 7_620.0, 133.755556, 0.0),
        Waypoint("descent1", 7_620.0, 133.755556, -7.62),
        Waypoint("hold", 1525, 102.88889, 0.0, hold_time=45 * 60, nominal=False),
        Waypoint("descent2", 1525, 102.88889, -7.62),
        Waypoint("approach", 304.8, 60.2, -3.1506),
        Waypoint("taxi\\landing", 0.0, 8.231111, 0.0, hold_time=10 * 60),
    ]
    pws = [
        Powerpoint("taxi\\TO", 0.28285, until_phase="takeoff"),
        Powerpoint("takeoff", 1.04, time=5 * 60),
        Powerpoint("climb", 0.95, until_phase="cruise"),
        Powerpoint("cruise", until_phase="descent1"),  # power filled by placeholder
        Powerpoint("descent1", 0.35, until_phase="hold"),
        Powerpoint("hold", until_phase="descent2"),    # power filled by placeholder
        Powerpoint("final", 0.35, until_phase="taxi\\landing"),
        Powerpoint("taxi\\landing", 0.28285, time=10 * 60),
    ]
    
    turboprop_H2 = Turboprop(
        name="PT6A-67D-H2",
        delta_mdot=4.9895161-4.5359237,
        mdot_min=4.5359237,
        eta_in=1,
        PI_comp=12.0,
        eta_comp=0.85,
        PI_cc=0.9733,
        eta_cc=0.97,
        LHV_fuel=119.96e6,
        T04=1274.15,
        eta_turb=0.85,
        eta_mech=0.95,
        c_pa=1005.0,
        k_air=1.4,
        c_pg=1157.4,
        k_gas=1.364729742
    )

    # if fc_split==0, substitute a “zero” fuel‐cell
    if fc_split == 0.0:
        class ZeroFuelCell:
            name = "none"
            power_req_max = 0.0
            throttle_TOGA = 0.0
            power_max_throttle = 0.0
            power_min = 0.0
            fc_mass = 0.0
            fc_volume = 0.0
            def get_TMS_values(self, power):
                # Qdot_fc, mdot_fc, mdot_air_in, mdot_air_out, mdot_H2O, mdot_recirculation
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        fc_model = ZeroFuelCell()
    else:
        fc_model = FuelCell(
            name="PEM-HT-FC",
            power_req_max=(TOGA*fc_split + delta_AP + base_AP),
            throttle_TOGA=throttle_TOGA
        )

    ac_model_H2 = Aircraft(
        name="H2-D2",
        wing_area=28.79,
        wing_span=17.64,
        CD0=0.024+CD_RAD,  # Add heat exchanger drag
        prop_diameter=2.78,
        eng=turboprop_H2,
        fc=fc_model,
        MTOW=MTOW,
        P_cc_min=0.057 * MAXC,  # [W] Minimum combustion chamber power
        delta_AP=delta_AP,  # [W] Thermal Management System power
    )
    
    mission_H2 = FlightMission(ac_model_H2, wps, pws, R_LHV=42.8/120, dt=dt, total_range_m=707e3, throttle_cruise=throttle_cruise)
    
    # Extract power loading and wing loading for corners of phases:
    takeoff = mission_H2.profile['phase'] == 'takeoff'
    climb   = (mission_H2.profile['phase'] == 'climb1') | (mission_H2.profile['phase'] == 'climb2')
    cruise  = mission_H2.profile['phase'] == 'cruise'
    hold    = mission_H2.profile['phase'] == 'hold'
    
    PW1_takeoff = mission_H2.profile['Pa'][takeoff][0]/(mission_H2.profile['mass'][takeoff][0]*G_0)
    WS1_takeoff = mission_H2.profile['mass'][takeoff][0]*G_0/mission_H2.ac.wing_area
    PW2_takeoff = mission_H2.profile['Pa'][takeoff][-1]/(mission_H2.profile['mass'][takeoff][-1]*G_0)
    WS2_takeoff = mission_H2.profile['mass'][takeoff][-1]*G_0/mission_H2.ac.wing_area
    
    PW1_climb   = mission_H2.profile['Pa'][climb][0]/(mission_H2.profile['mass'][climb][0]*G_0)
    WS1_climb   = mission_H2.profile['mass'][climb][0]*G_0/mission_H2.ac.wing_area
    PW2_climb   = mission_H2.profile['Pa'][climb][-1]/(mission_H2.profile['mass'][climb][-1]*G_0)
    WS2_climb   = mission_H2.profile['mass'][climb][-1]*G_0/mission_H2.ac.wing_area
    
    PW1_cruise  = mission_H2.profile['Pa'][cruise][0]/(mission_H2.profile['mass'][cruise][0]*G_0)
    WS1_cruise  = mission_H2.profile['mass'][cruise][0]*G_0/mission_H2.ac.wing_area
    PW2_cruise  = mission_H2.profile['Pa'][cruise][-1]/(mission_H2.profile['mass'][cruise][-1]*G_0)
    WS2_cruise  = mission_H2.profile['mass'][cruise][-1]*G_0/mission_H2.ac.wing_area
    
    PW1_hold    = mission_H2.profile['Pa'][hold][0]/(mission_H2.profile['mass'][hold][0]*G_0)
    WS1_hold    = mission_H2.profile['mass'][hold][0]*G_0/mission_H2.ac.wing_area
    PW2_hold    = mission_H2.profile['Pa'][hold][-1]/(mission_H2.profile['mass'][hold][-1]*G_0)
    WS2_hold    = mission_H2.profile['mass'][hold][-1]*G_0/mission_H2.ac.wing_area
    
    
    loading_points = np.array([
        [PW1_takeoff, WS1_takeoff, PW2_takeoff, WS2_takeoff],
        [PW1_climb, WS1_climb, PW2_climb, WS2_climb],
        [PW1_cruise, WS1_cruise, PW2_cruise, WS2_cruise],
        [PW1_hold, WS1_hold, PW2_hold, WS2_hold]
    ])
    
    # print(loading_points)
    
    H2_burnt = (mission_H2.profile['mass'][0] - mission_H2.profile['mass'][-1])/(1 - E_SCALE)
    # print(f"Total H2 mass burnt: {H2_burnt:.2f} kg")
    
    # Find the four most constraining TMS power points:
    # 1) Start of IDLE phase
    # 2) Start of TOGA phase
    # 3) Start of CRUISE phase
    # 4) Start of HOLD phase
    indexes = [
        np.where(mission_H2.profile['phase'] == 'taxi\\TO')[0][0],
        np.where(mission_H2.profile['phase'] == 'takeoff')[0][0],
        np.where(mission_H2.profile['phase'] == 'cruise')[0][0],
        np.where(mission_H2.profile['phase'] == 'hold')[0][0]
    ]
    # Reshape the TMS inputs to match the indexes
    TMS_inputs = {key: np.array([mission_H2.TMS_inputs[key][i] for i in indexes]) for key in mission_H2.TMS_inputs.keys()}

    fc_costs = np.array(fc_model.get_fc_cost(P_req_max=mission_H2.profile["P_fc"].max())) # easiest way to do it
                       
    # Determine the maximum fuel cell power across the three splits
    FC_outputs = dict(m_fc=fc_model.fc_mass,
                      V_fc=fc_model.fc_volume,
                      co2_fc=fc_model.fc_gwp,
                      fc_cost = fc_costs[0],
                      fc_stack_prod_cost = fc_costs[1],
                      fc_bop_cost = fc_costs[2],
                      fc_maintenance_cost = fc_costs[3],
                      fc_disposal_cost = fc_costs[4])
    # mission_H2.quicklook()  # Show quicklook plots
    
    # Get H2 mass, NOx mass, and max(mdot_NOx) for a nominal mission (- hold)
    # full‐length arrays
    no2_arr = mission_H2.profile['mdot_NOx']
    fuel_arr = mission_H2.profile['mdot_fuel']
    phase = mission_H2.profile['phase']

    # masks
    nominal_mask = phase != 'hold'
    toga_mask    = (phase == 'takeoff') | (phase == 'climb1') | (phase == 'climb2')
    cruise_mask  = (phase == 'cruise')  | (phase == 'descent1') | (phase == 'descent2')

    # integrate total masses over nominal
    m_NOx_nom = np.cumsum(no2_arr[nominal_mask])  * dt
    m_H2_nom  = np.cumsum(fuel_arr[nominal_mask]) * dt

    # peak NOx flow rates (only over nominal points in each phase)
    max_NOx_TO     = np.max(no2_arr[toga_mask   & nominal_mask])
    max_NOx_cruise = np.max(no2_arr[cruise_mask & nominal_mask])

    emissions_outputs = dict(
        m_NOx               = m_NOx_nom[-1],
        max_mdot_NOx_TO     = max_NOx_TO,
        max_mdot_NOx_cruise = max_NOx_cruise,
        m_H2_nom            = m_H2_nom[-1],
    )
    
    P_elmo = mission_H2.profile['P_fc'].max() - delta_AP
        
    return TMS_inputs, H2_burnt, FC_outputs, mission_H2.profile, loading_points, emissions_outputs, P_elmo


if __name__ == "__main__":
    
    mission_H2 = fpp_main(
        fc_split=0.33,
        throttle_TOGA=0.29,
        throttle_cruise=0.30,
        MTOW=7895.114629666458,
        CD_RAD=0.001013,
        delta_AP=160488.79,
        dt=0.1
    )
    
    print(mission_H2[5])
    
    # Add a cumulative FC/CC power split plot
    P_cc = mission_H2[3]['P_cc']/1e3  # Convert to kW
    P_fc = mission_H2[3]['P_fc']/1e3  # Convert to kW
    t    = mission_H2[3]['time']
    
    AP = 160488.79/1e3 + 2 * 8.4  # Auxiliary Power (kW)
    
    P_fc = P_fc + AP   # Add auxiliary power to fuel cell power

    # Mask to highlight non-nominal phases
    non_nominal = mission_H2[3]['phase'] == 'hold'

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Paired")
    fig, ax = plt.subplots(figsize=(8, 3))

    # Fuel cell fill
    ax.fill_between(
        t,
        0,
        P_fc,  # This should be P_fc, not P_cc for fuel cell
        color=palette[0],
        alpha=0.6,
        label="FC Power"
    )
    
    # Auxiliary power fill
    ax.fill_between(
        t,
        P_fc-AP,
        P_fc,
        color=palette[1],
        alpha=0.6,
        label="Auxiliary Power (AP)"
    )

    # Combustion chamber fill
    ax.fill_between(
        t,
        P_fc,  # Start from P_fc
        P_cc + P_fc,
        color=palette[4],
        alpha=0.6,
        label="CC Power"
    )

    # Plot the total power
    ax.plot(
        t,
        P_cc + P_fc,
        lw=1.3,
        color='k',
        label="Total Power"
    )

    # Find the start and end of the non-nominal phase
    if np.any(non_nominal):
        non_nominal_indices = np.where(non_nominal)[0]
        non_nominal_start = t[non_nominal_indices[0]]
        non_nominal_end = t[non_nominal_indices[-1]]
        
        # Add vertical lines at the start and end
        ax.axvline(non_nominal_start, color='r', linestyle='--')
        ax.axvline(non_nominal_end, color='r', linestyle='--')
        
        # Add annotation with arrow between the vertical lines
        y_arrow = 0.95 * (P_cc.max() + P_fc.max())
        
        ax.annotate(
            '',                               # no text here
            xy=(non_nominal_start, y_arrow),
            xytext=(non_nominal_end,  y_arrow),
            arrowprops=dict(arrowstyle='<->', color='r', lw=1.5)
        )

        x_mid = 0.5 * (non_nominal_start + non_nominal_end)
        ax.text(
            x_mid,
            y_arrow + 0.02 * y_arrow,         # nudge up 2 % of the y-span
            'Non-nominal Phase',
            ha='center',
            va='bottom',
            color='r',
            fontsize='medium',
            weight='bold'
        )

    # Set labels and title
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1.1 * (P_cc.max() + P_fc.max()))
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Flight Profile Power Split")
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    # print(mission_H2[-1])
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.animation import FuncAnimation, FFMpegWriter
    
    # from storage import tank

    # TOGA_vals   = np.linspace(0.1, 1.0, 21)
    # cruise_vals = np.linspace(0.1, 1.0, 21)
    # Tg, Cg      = np.meshgrid(TOGA_vals, cruise_vals)
    # mH2_grid    = np.zeros_like(Tg)

    # splits = np.linspace(0.0, 1.0, 21)

    # fig = plt.figure(figsize=(8,6))
    # ax  = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('TOGA throttle')
    # ax.set_ylabel('Cruise throttle')
    # ax.set_zlabel(r'H$_2$ \+ Tank \+ FC mass [kg]')
    
    # # ax.set_zlim(0, 1000)
    
    # ax.view_init(elev=30, azim=135)   # elevation=30deg, azimuth=135deg

    # def update(frame):
    #     s = splits[frame]
    #     # recompute mH2 for each (TOGA, cruise) at this split
    #     for i in range(Tg.shape[0]):
    #         for j in range(Tg.shape[1]):
    #             _, mH2, mFC, _ = fpp_main(
    #                 fc_split=s,
    #                 throttle_TOGA=Tg[i,j],
    #                 throttle_cruise=Cg[i,j],
    #                 MTOW=8037.6,
    #                 CD_HEX=0.0,
    #                 delta_AP=0.0,
    #                 dt=10
    #             )
    #             Mt, _, _, _, _, _, _ = tank.main_storage(mH2)
    #             mH2_grid[i, j] = mH2 + mFC['m_fc'] + Mt # Total mass of H2 + FC + tank mass
    #     ax.clear()
    #     ax.plot_surface(Tg, Cg, mH2_grid, cmap='viridis', edgecolor='none')
    #     ax.set_title(f"Power split = {s:.2f}")
    #     ax.set_xlabel('TOGA throttle')
    #     ax.set_ylabel('Cruise throttle')
    #     ax.set_zlabel(r'H$_2$ \+ Tank \+ FC mass [kg]')
    #     # ax.set_zlim(0, 1000)

    #     return []

    # ani = FuncAnimation(
    #     fig, update,
    #     frames=len(splits),
    #     interval=100,
    #     blit=False,
    # )
    
    # writer = FFMpegWriter(fps=10, bitrate=1800)
    # ani.save('figs/mh2_vs_split.mp4', writer=writer)

    # plt.tight_layout()
    # plt.show()
    
    
# if __name__ == "__main__":
#     # _, _ = fpp_main(fc_split=0.8, dt=10)

    
#     splits = np.linspace(0.0, 1.0, 101)
    
#     mH2 = np.zeros_like(splits)
#     mdot_dumpy = np.zeros_like(splits)
#     dumpy = np.zeros_like(splits)
    
#     throttle_TOGA = 0.9  # Throttle setting for TOGA phase
#     throttle_cruise = 0.4  # Throttle setting for cruise phase
    
#     for i, split in enumerate(splits):
#         print(f"Split {i+1}/{len(splits)}: {split:.2f}")
#         _, mH2[i], _, _ = fpp_main(fc_split=split, throttle_TOGA=throttle_TOGA, throttle_cruise=throttle_cruise, MTOW=8037.6, CD_HEX=0.0, delta_AP=0.0, dt=1)
#         # mH2[i], _, mdot_dumpy, _ = fpp_main(fc_split=split, dt=1)
#         # dumpy[i] = np.sum(mdot_dumpy) * 10
#         # print(f"mH2: {mH2:.2f} kg")

#     # plt.plot(splits, mH2)
#     plt.plot(splits, mH2)
#     plt.xlabel("Power Split [%]")
#     plt.ylabel("Mass of H2 Required [kg]")
#     plt.title(f"H2 Mass vs Power Split @ TOGA Throttle = {throttle_TOGA}, Cruise Throttle = {throttle_cruise}")
#     plt.tight_layout()
#     plt.show()