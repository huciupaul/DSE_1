# file: performance/MTOW/mtow.py
# desc: Check MTOW against original aircraft constraints analysis and return a score.

# Global imports
import numpy as np
from typing import Tuple, Union

# Local imports
from global_constants import MTOW_orig, C_d_0_orig
from performance.MTOW import ca
 
_cons      = ca.constraint_curves(C_d_0_orig, MTOW_orig) 
_WS_x      = _cons["x_axis"].ravel()
_PW_cl     = _cons["PW_cl"]
_PW_cr     = _cons["PW_cr"]
_WS_stall  = _cons["WS_stall"]
# Original 

_PW_req_curve_TO = np.maximum(_PW_cl, _PW_cr).ravel()
_PW_req_curve_cr = _PW_cr.ravel()

def power_and_wing_loading(
    loading: np.ndarray
    ) -> np.ndarray:
    """Check loading against original aircraft constraints analysis and return a mask.

    Args:       
        loading (np.ndarray): Mass loading tensor with shape (N, M, P, 4, 4) where:
            N: Power splits (FC use vs Turboprop use)
            M: FC TOGA throttle setting
            P: FC cruise throttle setting
            4: Flight-phase dependent loading vector:
                < PW_1, WS_1, PW_2, WS_2 > x 4
                where:
                    PW_1: Power loading at phase start
                    WS_1: Wing loading at phase start
                    PW_2: Power loading at phase end
                    WS_2: Wing loading at phase end
    
    Returns:
        np.ndarray: Boolean mask indicating whether the tensor indices meet the constraints.
    """
    # We receive a 5D tensor (N, M, P, 4, 4) and we need to check the constraints
    # for each flight phase (4 phases) and return a mask of NxMxP shape
    out_shape = loading.shape[:-2]  # Shape (N, M, P)
    
    # Extract the 4 flight phases
    PW_1_TO = loading[..., :2, 0]
    WS_1_TO = loading[..., :2, 1]
    PW_2_TO = loading[..., :2, 2]
    WS_2_TO = loading[..., :2, 3]
    
    PW_1_cr = loading[..., 2:, 0]
    WS_1_cr = loading[..., 2:, 1]
    PW_2_cr = loading[..., 2:, 2]
    WS_2_cr = loading[..., 2:, 3]
    
    # Reshape the arrays to be (NxMxP, 2)
    PW_1_TO = PW_1_TO.reshape(-1, 2)
    WS_1_TO = WS_1_TO.reshape(-1, 2)
    PW_2_TO = PW_2_TO.reshape(-1, 2)
    WS_2_TO = WS_2_TO.reshape(-1, 2)
    PW_1_cr = PW_1_cr.reshape(-1, 2)
    WS_1_cr = WS_1_cr.reshape(-1, 2)
    PW_2_cr = PW_2_cr.reshape(-1, 2)
    WS_2_cr = WS_2_cr.reshape(-1, 2)

    # Interpolate and check the constraints for each flight phase
    # Take-off phase
    mask_TO = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.interp(WS_1_TO[:, 0], _WS_x, _PW_req_curve_TO) <= PW_1_TO[:, 0],
                np.interp(WS_2_TO[:, 0], _WS_x, _PW_req_curve_TO) <= PW_2_TO[:, 0]
                ),
            np.logical_and(
                np.interp(WS_1_TO[:, 1], _WS_x, _PW_req_curve_TO) <= PW_1_TO[:, 1],
                np.interp(WS_2_TO[:, 1], _WS_x, _PW_req_curve_TO) <= PW_2_TO[:, 1]
            )
        ),
        np.logical_and(
            np.logical_and(
                WS_1_TO[:, 0] <= _WS_stall,
                WS_2_TO[:, 0] <= _WS_stall
                ),
            np.logical_and(
                WS_1_TO[:, 1] <= _WS_stall,
                WS_2_TO[:, 1] <= _WS_stall
            )
        )
    )
    
    
    
    # Cruise phase
    mask_cr = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.interp(WS_1_cr[:, 0], _WS_x, _PW_req_curve_cr) <= PW_1_cr[:, 0],
                np.interp(WS_2_cr[:, 0], _WS_x, _PW_req_curve_cr) <= PW_2_cr[:, 0]
            ),
            np.logical_and(
                np.interp(WS_1_cr[:, 1], _WS_x, _PW_req_curve_cr) <= PW_1_cr[:, 1],
                np.interp(WS_2_cr[:, 1], _WS_x, _PW_req_curve_cr) <= PW_2_cr[:, 1]
            )
        ),
        np.logical_and(
            np.logical_and(
                WS_1_cr[:, 0] <= _WS_stall,
                WS_2_cr[:, 0] <= _WS_stall
            ),
            np.logical_and(
                WS_1_cr[:, 1] <= _WS_stall,
                WS_2_cr[:, 1] <= _WS_stall
            )
        )
    )
    # Reshape the masks back to the original shape
    mask_TO = mask_TO.reshape(out_shape)
    mask_cr = mask_cr.reshape(out_shape)
    
    # Combine the masks for both flight phases
    mask = np.logical_and(mask_TO, mask_cr)

    return mask_TO