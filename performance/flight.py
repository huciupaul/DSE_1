import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Local imports
from common.constants import MAXC, G_0, A_inlet
from common.atmosphere import isa_atmosphere
from turboprop import Turboprop

# Dataclass definitions
@dataclass
class Aircraft:
    name: str
    wing_area: float
    wing_span: float
    CD0: float
    prop_diameter: float
    eng: Turboprop
    MTOW: float
    MAXC: float = MAXC
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

def mass_adjustment(mdot_fuel: np.ndarray, t: np.ndarray, R_eta_th: np.ndarray, R_LHV) -> float:
    """Get the H2 mass required to achieve the same thrust as the JA1 engine."""
    mdot_H2 = mdot_fuel * R_eta_th * R_LHV
    m_H2 = np.trapezoid(mdot_H2, t)  # Integrate fuel flow over time
    return m_H2
    
class FlightMission:
    """Time-resolved kinematic + power mission profile.

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
                 cruise_wp_name: str = "cruise") -> None:
        if len(waypoints) < 2:
            raise ValueError("Need at least two waypoints.")
        self.ac = aircraft
        self.wps = waypoints
        self.pws = powerpoints
        self.R_LHV = R_LHV
        self.dt = dt
        self.total_range = total_range_m
        self.cruise_wp_name = cruise_wp_name.lower()
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
        T_arr, P_arr, rho_arr, a_arr = isa_atmosphere(h_arr)
        Pr_arr = np.zeros_like(V_arr)
        Pa_arr = np.zeros_like(V_arr)
        m_arr  = np.zeros_like(V_arr);   m_arr[0] = self.ac.MTOW
        mdot_fuel_arr = np.zeros_like(V_arr)
        mdot_air_arr  = np.zeros_like(V_arr)
        eta_th_arr    = np.zeros_like(V_arr)
        eta_prop_arr  = np.zeros_like(V_arr)

        for pp, sl in self.__pp_slicer(phase_arr, time_arr):
            if pp.power is not None:
                # Powerpoint with power
                Pr_arr[sl] = pp.power*self.ac.MAXC
                # Propulsive efficiency
                Th = Pr_arr[sl]/V_arr[sl]
                
                # Fuel mass flow using Torenbeek method (PSFC)
                mdot_fuel_arr[sl], eta_th_arr[sl], eta_prop_arr[sl] = self.ac.eng.compute(
                    T_arr[sl], P_arr[sl], rho_arr[sl], V_arr[sl], R_LHV=self.R_LHV, Pr=Pr_arr[sl]
                )
                
                # Available power
                Pa_arr[sl] = Pr_arr[sl]/eta_prop_arr[sl]
                
                m_start = m_arr[sl.start - 1] if sl.start > 0 else self.ac.MTOW
                burn = np.cumsum(mdot_fuel_arr[sl]) * self.dt          # cumulative fuel burnt _within_ slice
                m_arr[sl] = m_start - np.concatenate(([0.0], burn[:-1]))
                mdot_air_arr[sl] = rho_arr[sl] * V_arr[sl] * A_inlet
        
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
                    
                    mdot_fuel, eta_th, eta_prop = self.ac.eng.compute(
                        T_arr[i], P_arr[i], rho_arr[i], V_arr[i]
                    )
                    
                    Pa = Pr/eta_prop
                    m_arr[i+1] = m_arr[i] - mdot_fuel*self.dt
                    
                    # 7) Store results @current time step
                    Pr_arr[i] = Pr
                    Pa_arr[i] = Pa
                    mdot_fuel_arr[i] = mdot_fuel
                    mdot_air_arr[i] = rho_arr[i] * V_arr[i] * A_inlet
                    eta_th_arr[i] = eta_th
                    eta_prop_arr[i] = eta_prop     

        # finally
        self._profile = dict(time=time_arr, V=V_arr, alt=h_arr, ROC=roc_arr,
                            phase=phase_arr, Pa=Pa_arr, Pr=Pr_arr,
                            mdot_fuel=mdot_fuel_arr, mdot_air=mdot_air_arr,
                            eta_th=eta_th_arr, eta_prop=eta_prop_arr,
                            mass=m_arr)

    # ---------------------------------------------------------------------
    #  QUICK‑LOOK PLOTS (kinematics + power)
    # ---------------------------------------------------------------------
    def quicklook(self) -> None:
        import matplotlib.pyplot as plt
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
        # for ax in axs:
        #     ax.set_xlabel("Time [s]")
        #     ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

    
if __name__ == "__main__":
    wps = [
        Waypoint("taxi\\TO", 0.0, 8.231111, 0.0, hold_time=10 * 60),
        Waypoint("takeoff", 0.0, 54.0167, 797 / 60),
        Waypoint("climb1", 2_438.4, 61.73333, 797 / 60),
        Waypoint("climb2", 4_876.8, 61.73333, 797 / 60),
        Waypoint("cruise", 7_620.0, 142.501, 0.0),
        Waypoint("descent1", 7_620.0, 142.501, -7.62),
        Waypoint("hold", 450.0, 102.88889, 0.0, hold_time=45 * 60, nominal=True),
        Waypoint("descent2", 450.0, 102.88889, -7.62),
        Waypoint("approach", 304.8, 60.2, -3.1506),
        Waypoint("taxi\\landing", 0.0, 8.231111, 0.0, hold_time=10 * 60),
    ]

    pws = [
        Powerpoint("taxi\\TO", 0.1415, until_phase="takeoff"),
        Powerpoint("takeoff", 1.04, time=5 * 60),
        Powerpoint("climb", 0.85, until_phase="cruise"),
        Powerpoint("cruise", until_phase="descent1"),  # power filled by placeholder
        Powerpoint("descent1", 0.50, until_phase="hold"),
        Powerpoint("hold", until_phase="descent2"),    # power filled by placeholder
        Powerpoint("final", 0.50, until_phase="taxi\\landing"),
        Powerpoint("taxi\\landing", 0.1415, time=10 * 60),
    ]
    
    turboprop_JA1 = Turboprop(
        name="PT6A-67D",
        eta_in=1,
        PI_comp=12.0,
        eta_comp=0.85,
        PI_cc=0.95,
        eta_cc=0.97,
        LHV_fuel=42.8e6,
        T04=1274.15,
        eta_turb=0.85,
        eta_mech=0.95,
        c_pa=1005.0,
        k_air=1.4,
        c_pg=1_150.0,
        k_gas=1.33
    )
    
    turboprop_H2 = Turboprop(
        name="PT6A-67D-H2",
        eta_in=1,
        PI_comp=12.0,
        eta_comp=0.85,
        PI_cc=0.9733,
        eta_cc=0.995,
        LHV_fuel=119.96e6,
        T04=1274.15,
        eta_turb=0.89,
        eta_mech=0.99,
        c_pa=1005.0,
        k_air=1.4,
        c_pg=1157.4,
        k_gas=1.364729742
    )

    ac_model_JA1 = Aircraft(
        name="Beechcraft 1900D",
        wing_area=28.79,
        wing_span=17.64,
        CD0=0.0215,
        prop_diameter=2.78,
        eng=turboprop_JA1,
        MTOW=7_765.0,
    )
    
    ac_model_H2 = Aircraft(
        name="Beechcraft 1900D-H2",
        wing_area=28.79,
        wing_span=17.64,
        CD0=0.023,
        prop_diameter=2.78,
        eng=turboprop_H2,
        MTOW=7_765.0,
    )

    mission_JA1 = FlightMission(ac_model_JA1, wps, pws, R_LHV=1.0, dt=0.5, total_range_m=2_368e3)
    # mission_H2_dummy = FlightMission(ac_model_H2, wps, pws, dt=0.5, total_range_m=1311e3)
    mission_H2 = FlightMission(ac_model_H2, wps, pws, R_LHV=42.8/120, dt=0.5, total_range_m=707e3)
    m_start = mission_JA1.profile['mass'][0]
    m_fin = mission_JA1.profile['mass'][-1]
    print(f"Aircraft: {ac_model_JA1.name}")
    print(f"Engine: {ac_model_JA1.eng.name}")
    print(f"Initial mass: {m_start:.1f} kg")
    print(f"Final mass: {m_fin:.1f} kg")
    print(f"Fuel burn: {m_start - m_fin:.1f} kg")
        
    # m_H2_cc = mass_adjustment(mission_JA1.profile['mdot_fuel'], mission_JA1.profile['time'], 
    #                     ac_model_JA1.eng.LHV_fuel/ac_model_H2.eng.LHV_fuel,
    #                     np.divide(mission_JA1.profile['eta_th'], mission_H2_dummy.profile['eta_th']))
    
    # m_H2_fc = mass_adjustment(mission_JA1.profile['mdot_fuel'], mission_JA1.profile['time'],
    #                     ac_model_JA1.eng.LHV_fuel/ac_model_H2.eng.LHV_fuel,
    #                     mission_JA1.profile['eta_th']/0.7)
    
    # print(f"H2 mass, pure combustion: {m_H2_cc:.1f} kg")
    # print(f"H2 mass, solely fuel cell: {m_H2_fc:.1f} kg")
    # # print(np.min(mission_JA1.profile['mdot_air'])/2, np.max(mission_JA1.profile['mdot_air'])/2)
    m_start = mission_H2.profile['mass'][0]
    m_fin = mission_H2.profile['mass'][-1]
    burn_cc = m_start - m_fin
    
    R_eta_th = mission_H2.profile['eta_th']/0.7
    burn_fc = mass_adjustment(mission_H2.profile['mdot_fuel'], mission_H2.profile['time'],
                        1, R_eta_th)
    
    print("----")
    print(f"Aircraft: {ac_model_H2.name}")
    print(f"Engine: {ac_model_H2.eng.name}")
    print(f"Initial mass: {m_start:.1f} kg")
    print(f"Final mass: {m_fin:.1f} kg")
    print(f"Fuel burn (CC only): {burn_cc:.1f} kg")
    print(f"Fuel burn (FC only): {burn_fc:.1f} kg")
    
    mission_JA1.quicklook()
    mission_H2.quicklook()
