import numpy as np
from typing import Tuple, Union

from global_constants import Beechcraft_1900D, seat_pitch, rho_cargo, l_aft_cyl, w_aft_cyl, h_aft_cyl_ave, d_aft_cone_beg, d_aft_cone_end, l_aft_cone, X_aft_cone_beg, X_aft_cone_end


def determine_position_in_cone(
    L_tank: Union[float, np.ndarray],
    d_tank: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    a = (
        X_aft_cone_beg
        - X_aft_cone_end
    ) / (
        d_aft_cone_beg
        - d_aft_cone_end
    )
    b = X_aft_cone_beg - a * d_aft_cone_beg

    X_tank_back = a * d_tank + b
    X_tank_front = X_tank_back - L_tank

    return X_tank_front, X_tank_back

def determine_diameter_in_cone(
    X_position: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    a = (
        X_aft_cone_beg
        - X_aft_cone_end
    ) / (
        d_aft_cone_beg
        - d_aft_cone_end
    )
    b = X_aft_cone_beg - a * d_aft_cone_beg
    return (X_position - b) / a

def check_for_seating_interference(
    X_tank_front: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
    Xc0 = Beechcraft_1900D["X_aft_cyl_beg"]
    delta = Xc0 - X_tank_front

    rows_removed = np.ceil(delta / seat_pitch)
    num_PAX = Beechcraft_1900D["num_PAX"] - 2 * rows_removed

    # if delta <= 0, no seats removed
    num_PAX = np.where(delta > 0, num_PAX, Beechcraft_1900D["num_PAX"])
    X_aft_cyl_beg = np.where(
        delta > 0,
        Xc0 - rows_removed * seat_pitch,
        Xc0
    )
    return X_aft_cyl_beg, num_PAX

def shift_tank_fwd(
    X_tank_front: Union[float, np.ndarray],
    X_tank_back: Union[float, np.ndarray],
    X_aft_cyl_beg: Union[float, np.ndarray],
    L_tank: Union[float, np.ndarray],
    d_tank: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = X_aft_cyl_beg < X_tank_front

    new_front = np.where(mask, X_aft_cyl_beg, X_tank_front)
    new_back  = np.where(mask, new_front + L_tank, X_tank_back)
    new_d     = np.where(mask, determine_diameter_in_cone(new_back), d_tank)

    return new_front, new_back, new_d

def tank_and_TMS_positioning_specs(
    L_tank: Union[float, np.ndarray],
    d_tank_TMS: Union[float, np.ndarray],
    X_aft_cyl_beg: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lc0 = l_aft_cyl
    L_tank_cyl = np.where(Lc0 >= L_tank, Lc0, L_tank)
    L_tank_cone = L_tank - L_tank_cyl

    # cylindrical part
    V_cyl = (
        L_tank_cyl
        * h_aft_cyl_ave / 2
        * w_aft_cyl / 2
        * np.pi
    )
    X_cyl = X_aft_cyl_beg + L_tank_cyl / 2

    # conical part
    d0 = d_aft_cone_beg
    V_cone = (
        1/3
        * np.pi
        * L_tank_cone
        * (d0**2 + d_tank_TMS**2 + d0 * d_tank_TMS) / 4
    )
    X_cone = X_aft_cone_beg + (
        L_tank_cone / 4
        * (
            X_aft_cone_beg**2
            + 2 * X_aft_cone_beg * d_tank_TMS
            + 3 * d_tank_TMS**2
          )
        / (
            X_aft_cone_beg**2
            + X_aft_cone_beg * d_tank_TMS
            + d_tank_TMS**2
          )
    )

    V_tank_TMS = V_cyl + V_cone
    X_tank_TMS = (V_cyl * X_cyl + V_cone * X_cone) / V_tank_TMS

    return L_tank_cone, d_tank_TMS, V_tank_TMS, X_tank_TMS

def calculate_cargo_specs(
    L_tank_cone: Union[float, np.ndarray],
    d_cargo_front: Union[float, np.ndarray],
    X_aft_cyl_beg: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L0 = l_aft_cone
    L_aft_cargo = L0 - L_tank_cone
    d_end = d_aft_cone_end

    V_aft_cargo = (
        1/3
        * np.pi
        * L_aft_cargo
        * (d_cargo_front**2 + d_end**2 + d_cargo_front * d_end) / 4
    )
    M_aft_cargo = rho_cargo * V_aft_cargo
    X_aft_cargo = X_aft_cyl_beg + (
        L_aft_cargo / 4
        * (
            d_cargo_front**2
            + 2 * d_cargo_front * X_aft_cone_end
            + 3 * X_aft_cone_end**2
          )
        / (
            d_cargo_front**2
            + d_cargo_front * X_aft_cone_end
            + X_aft_cone_end**2
          )
    )

    return V_aft_cargo, M_aft_cargo, X_aft_cargo

def cargo_main(
    L_tank: Union[float, np.ndarray],
    d_tank: Union[float, np.ndarray]
) -> dict:
    Xf, Xb                  = determine_position_in_cone(L_tank, d_tank)
    Xc, num_PAX             = check_for_seating_interference(Xf)
    Xf, Xb, d_tank_TMS      = shift_tank_fwd(Xf, Xb, Xc, L_tank, d_tank)
    Lc, d_tank_TMS, Vt, Xt  = tank_and_TMS_positioning_specs(L_tank, d_tank_TMS, Xc)
    Va, Ma, Xa              = calculate_cargo_specs(Lc, d_tank_TMS, Xc)

    return {
        "X_tank_front": Xf,
        "X_tank_back":  Xb,
        "X_tank_TMS":   Xt,
        "V_tank_TMS":   Vt,
        "X_aft_cargo":  Xa,
        "M_aft_cargo":  Ma,
        "V_aft_cargo":  Va,
        "num_PAX":      num_PAX
    }