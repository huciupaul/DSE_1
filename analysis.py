# file: analysis.py
# desc: This script analyzes the 4D design space tensor of the SHTARWaRS design tool.

# Global imports
import numpy as np
import pickle
import copy
from typing import Tuple

# Local imports
from global_constants import Beechcraft_1900D
from performance.MTOW.mtow import power_and_wing_loading as pws
from performance.Integration.integration import main as cg_excursion

# Cached constants
M_CARGO_AFT = Beechcraft_1900D["M_cargo_aft"]  # [kg] Cargo mass in the aft compartment

def load_tensor(file_path: str) -> np.ndarray:
    """
    Load the 4D design space tensor from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file containing the design space tensor.
        
    Returns:
        np.ndarray: The loaded 4D design space tensor.
    """
    with open(file_path, 'rb') as f:
        tensor = pickle.load(f)
    return tensor

def f_obj(design: np.ndarray) -> np.ndarray:
    """Design tensor objective function to evaluate the design based.

    Args:
        design (np.ndarray): Design tensor with shape (N, M, P, Q) where:
            N: Power splits (FC use vs Turboprop use)
            M: FC TOGA throttle setting
            P: FC cruise throttle setting
            Q: Design variables: 
                < m_EPS, m_FC, m_H2, m_storage, m_TMS, V_FC, V_storage, V_ELMO, MTOW, L_storage, D_storage >

    Returns:
        np.ndarray: A 3D tensor of scores with shape (N, M, P) where each element is the score for the corresponding design.
    """
    pass

def main(weights: tuple=(0.5, 0.25, 0.125, 0.125),
         dim_bounds: np.ndarray=np.array([[0.1, 1.0],
                                          [0.1, 1.0],
                                          [0.0, 1.0]]),
         fpath_design: str='data/logs/result_tensor.pkl',
         fpath_loading: str='data/logs/loading_tensor.pkl') -> None:
    """Main function to collapse the 4D design space to a 3D scalar field using the provided weights.

    Args:
        weights (tuple, optional): Integration, MTOW, N_PAX, and m_cargo weights. Defaults to (0.5, 0.25, 0.125, 0.125).
        dim_bounds (np.ndarray, optional): Bounds for the dimensions of the design space tensor. Defaults to [[0.1, 1.0], [0.1, 1.0], [0.0, 1.0]].
        fpath (str, optional): Filepath to the design space tensor created from main.py. Defaults to 'data/design_space.pkl'.
    """
    # Quick validation of weights
    if sum(weights) != 1.0:
        raise ValueError("Weights must sum to 1.0")
    if len(weights) != 4:
        raise ValueError("Weights must be a tuple of length 4")
    
    # Load the 4D design space tensor
    try:
        tensor = load_tensor(fpath_design)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fpath_design}")
    
    # Load the 4D loading space tensor
    try:
        loading = load_tensor(fpath_loading)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fpath_loading}")
    
    print(f"Design space tensor shape: {tensor.shape}")
    print(f"Loading space tensor shape: {loading.shape}")
    
    # Create axes for the tensor dimensions
    N, M, P, Q = tensor.shape
    splits = np.linspace(dim_bounds[0, 0], dim_bounds[0, 1], N)
    toga_throttle = np.linspace(dim_bounds[1, 0], dim_bounds[1, 1], M)
    cruise_throttle = np.linspace(dim_bounds[2, 0], dim_bounds[2, 1], P)
        
    ### TENSOR SHAPE DESCRIPTION ###
    # The tensor is expected to have the shape (N, M, P, Q) where:
    # N: Power splits (FC use vs Turboprop use)
    # M: FC TOGA throttle setting
    # P: FC cruise throttle setting
    # Q: Design variables: 
    #   < m_EPS, m_FC, m_H2, m_storage, m_TMS, V_FC, V_storage, V_ELMO, MTOW, L_storage, D_storage >
    # Keep a 'design' tensor that maintains the original tensor structure
    design      = copy.deepcopy(tensor)
    # Initialize a 'tradeoff' tensor with the same N, M, P dimensions and a 4th dimension containing 4 np.nan values

    # Apply design space constraints
    valid_integration, N_PAX, m_cargo = cg_excursion(design)
    valid_mass = pws(loading)
    
    valid3d = valid_integration & valid_mass
    # Keep only valid designs, fill rest with np.nan
    # lift into a 4th dim so it matches (N, M, P, Q)
    valid4d = valid3d[..., None]        # shape (N, M, P, 1)
    valid4d = np.broadcast_to(valid4d, design.shape)

    # now prune
    design[~valid4d] = np.nan
    
    print(f"Percentage pruned by integration: {np.sum(~valid_integration) / np.prod(design.shape[:-1]) * 100:.2f}%")
    print(f"Percentage pruned by mass loading: {np.sum(~valid_mass) / np.prod(design.shape[:-1]) * 100:.2f}%")
    print(f"Total percentage pruned from design space: {np.sum(~valid4d) / np.prod(design.shape) * 100:.2f}%")
    
    # Calculate scores for each design configuration
    # scores = f_obj(design)
    
    # # Find optimal design configuration
    # max_score_idx = np.unravel_index(np.nanargmax(scores, axis=None), scores.shape)
    # optimal_design = design[max_score_idx]
    # optimal_setting = {
    #     'power_split': splits[max_score_idx[0]],
    #     'toga_throttle': toga_throttle[max_score_idx[1]],
    #     'cruise_throttle': cruise_throttle[max_score_idx[2]],
    #     'design_variables': optimal_design
    # }
    
    # # Print the optimal design configuration
    # print("Optimal Design Configuration:")
    # print(f"Power Split: {optimal_setting['power_split']:.2f}")
    # print(f"TOGA Throttle Setting: {optimal_setting['toga_throttle']:.2f}")
    # print(f"Cruise Throttle Setting: {optimal_setting['cruise_throttle']:.2f}")
    # print(f"Design Variables: {optimal_setting['design_variables']}")
    # print(f"Maximum Score: {np.nanmax(scores):.4f}")
    # print(f"Optimal Design Index: {max_score_idx}")
    
if __name__ == "__main__":
    main(
        weights=(0.5, 0.25, 0.125, 0.125),
        dim_bounds=np.array([[0.1, 1.0],
                                [0.1, 1.0],
                                [0.0, 1.0]]),
        fpath_design='data/logs/result_tensor.pkl',
        fpath_loading='data/logs/loading_tensor.pkl'
    )