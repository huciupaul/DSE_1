import numpy as np
import pickle
import copy
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Local imports
from global_constants import NOx_pPax_TO, NOx_pPax_cruise
from performance.MTOW.mtow import power_and_wing_loading as pws
from performance.Integration.integration import main as cg_excursion
from cost import calc_cost as f_cost
from sus import get_sus as f_gwp

# Load data once when app starts
print("Loading tensors...")
with open('data/logs/result_tensor.pkl', 'rb') as f:
    design_tensor = pickle.load(f)

with open('data/logs/loading_tensor.pkl', 'rb') as f:
    loading_tensor = pickle.load(f)
    
with open('data/logs/CostnSus.pkl', 'rb') as f:
    cost_gwp_tensor = pickle.load(f)

# Pre-calculate constraint masks
print("Calculating constraint masks...")
valid_integration, N_PAX, m_cargo = cg_excursion(design_tensor)
valid_mass = pws(loading_tensor)
valid_NOx = (design_tensor[..., 15]/N_PAX < NOx_pPax_TO) & (design_tensor[..., 16]/N_PAX < NOx_pPax_cruise)

# Define dimension bounds
dim_bounds = np.array([[0.1, 0.9], [0.1, 1.0], [0.1, 1.0]])
N, M, P, Q = design_tensor.shape
power_splits = np.linspace(dim_bounds[0, 0], dim_bounds[0, 1], N)
toga_throttle = np.linspace(dim_bounds[1, 0], dim_bounds[1, 1], M)
cruise_throttle = np.linspace(dim_bounds[2, 0], dim_bounds[2, 1], P)

# Initialize the dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Design Space Visualization", style={'textAlign': 'center', 'margin': '10px'}),
    
    # Controls row - compact at the top
    html.Div([
        html.Div([
            html.Label("Cost Weight:"),
            dcc.Slider(
                id='weight-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                marks={i/10: str(i/10) for i in range(11)}
            ),
        ], style={'width': '60%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Checklist(
                id='constraint-checklist',
                options=[
                    {'label': 'Integration Constraint', 'value': 'integration'},
                    {'label': 'Mass Constraint', 'value': 'mass'},
                    {'label': 'NOx Constraint', 'value': 'nox'}
                ],
                value=['integration', 'mass', 'nox'],
                inline=True
            ),
        ], style={'width': '40%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '10px'}),
    
    # Main content area - horizontal layout with 3 sections
    html.Div([
        # Left panel - Optimal design details
        html.Div([
            html.H3("Optimal Design Details", style={'marginTop': '0'}),
            html.Pre(id='design-info', style={
                'backgroundColor': '#f8f9fa', 
                'padding': '10px',
                'borderRadius': '5px',
                'height': 'calc(100vh - 200px)',
                'overflowY': 'auto'
            })
        ], style={
            'width': '25%', 
            'padding': '10px',
            'boxSizing': 'border-box'
        }),
        
        # Middle panel - 3D graph
        html.Div([
            dcc.Graph(
                id='3d-scatter',
                style={'height': 'calc(100vh - 150px)'},
                config={'displayModeBar': True}
            )
        ], style={
            'width': '75%',
            'boxSizing': 'border-box'
        })
    ], style={
        'display': 'flex',
        'flexWrap': 'nowrap',
        'height': 'calc(100vh - 150px)'
    })
])

# The callback function update_graph needs to be modified:
@app.callback(
    [Output('3d-scatter', 'figure'),
     Output('design-info', 'children')],
    [Input('weight-slider', 'value'),
     Input('constraint-checklist', 'value')]
)
def update_graph(weight, constraints):
    # Apply selected masks - same as before
    valid3d = np.ones(design_tensor.shape[:-1], dtype=bool)
    if 'integration' in constraints:
        valid3d = valid3d & valid_integration
    if 'mass' in constraints:
        valid3d = valid3d & valid_mass
    if 'nox' in constraints:
        valid3d = valid3d & valid_NOx
    
    # Rest of the processing - same as before until the voxel creation
    # [code for calculating scalar_field, viz_field, finding optimal design]
    
    # Calculate cost and GWP components
    design = copy.deepcopy(design_tensor)
    valid4d = valid3d[..., None]
    valid4d = np.broadcast_to(valid4d, design.shape)
    design[~valid4d] = np.nan
    
    # Calculate cost and GWP components
    GWP_fc = design[..., 20]  # GWP of fuel cell production/disposal
    GWP_sto = design[..., 21]  # GWP of storage system production/disposal
    GWP_eps = design[..., 22]  # GWP of EPS production/disposal
    m_h2_nom = design[..., 18]   # Nominal mass of hydrogen used
    m_nox = design[..., 14]  # Mass of NOx produced
    
    fc_cost = design[..., 17]  # Cost of the fuel cell
    P_eps = design[..., 19]  # Max power used by electric propulsion system
    m_sto = design[..., 3]  # Mass of the storage system
    m_h2 = design[..., 2]  # Total mass of hydrogen stored
    
    # Calculate GWP and cost
    GWP = f_gwp(GWP_fc, GWP_sto, GWP_eps, m_h2_nom, m_nox) / N_PAX
    cost = f_cost(fc_cost, P_eps, m_sto, m_h2) / N_PAX
    
    # Normalize values
    valid_gwp = ~np.isnan(GWP)
    valid_cost = ~np.isnan(cost)
    if np.any(valid_gwp):
        GWP_norm = GWP / np.nanmax(GWP)
    else:
        GWP_norm = np.zeros_like(GWP)
    
    if np.any(valid_cost):
        cost_norm = cost / np.nanmax(cost)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Calculate scalar field
    scalar_field = weight * cost_norm + (1 - weight) * GWP_norm
    
    # Calculate 1/(1+score) for visualization (higher value is better)
    viz_field = 1 / (1 + scalar_field)
    
    # Find optimal design
    min_value = np.nanmin(scalar_field)
    if not np.isnan(min_value):
        min_idx = np.where(scalar_field == min_value)
        opt_idx = (min_idx[0][0], min_idx[1][0], min_idx[2][0])
        optimal_design = design_tensor[opt_idx]
        performance_metrics = cost_gwp_tensor[opt_idx]
        
        # Prepare optimal design info text as before...
        design_info = [
            f"Optimal Design Score: {min_value:.4f}:",
            f"Power Split: {power_splits[opt_idx[0]]:.3f}",
            f"TOGA Throttle: {toga_throttle[opt_idx[1]]:.3f}",
            f"Cruise Throttle: {cruise_throttle[opt_idx[2]]:.3f}",
            f"\nPerformance Metrics:",
            f"Lifetime cost: {performance_metrics[0]/1e6:.2f} M€/PAX",
            f"GWP: {performance_metrics[1]:.2f} kg CO₂e/FLIGHT/PAX",
            "\nDesign Parameters:",
            f"m_EPS: {optimal_design[0]:.5f} kg",
            f"m_FC: {optimal_design[1]:.5f} kg",
            f"m_H2: {optimal_design[2]:.5f} kg",
            f"m_storage: {optimal_design[3]:.5f} kg",
            f"m_TMS_total: {np.sum(optimal_design[11:14]):.5f} kg",
            f"m_TMS_front: {optimal_design[11]:.5f} kg",
            f"m_TMS_aft: {optimal_design[12]:.5f} kg",
            f"m_TMS_mid: {optimal_design[13]:.5f} kg",
            f"V_FC: {optimal_design[4]:.5f} m³",
            f"V_storage: {optimal_design[5]:.5f} m³",
            f"V_ELMO: {optimal_design[6]:.5f} m³",
            f"MTOW: {optimal_design[7]:.5f} kg\n"
            f"m_NOx: {optimal_design[14]:.5f} kg"
        ]
        design_info_str = "\n".join(design_info)
    else:
        opt_idx = None
        design_info_str = "No valid designs found with current constraints."
        
    # Create the figure
    fig = go.Figure()
    
    # Instead of volume, use scatter3d with visible cubes
    if np.any(valid3d):
        # Get indices where we have valid data
        x_idx, y_idx, z_idx = np.where(valid3d)
        
        # Get scores for these points
        scores = viz_field[x_idx, y_idx, z_idx]
        
        # Normalize scores for sizing and coloring
        if len(scores) > 0:
            max_score = np.max(scores)
            if max_score > 0:
                norm_scores = scores / max_score
            else:
                norm_scores = scores
        else:
            norm_scores = []
        
        # Convert indices to actual coordinate values
        x_coords = power_splits[x_idx]
        y_coords = toga_throttle[y_idx] 
        z_coords = cruise_throttle[z_idx]
        
        # Create cube markers with size based on score
        marker_sizes = 10 + 20 * norm_scores  # Size between 10 and 30 based on score
        
        # Add scatter3d with cube markers
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=norm_scores,
                colorscale='Viridis',
                opacity=0.7,
                symbol='square',  # Using square for a more cube-like appearance
                colorbar=dict(title='1/(1+score)\n',
                              ypad=100)
            ),
            name='Design Space'
        ))

    # Add optimal design point if it exists (same as before)
    if opt_idx is not None:
        opt_point = go.Scatter3d(
            x=[power_splits[opt_idx[0]]],
            y=[toga_throttle[opt_idx[1]]],
            z=[cruise_throttle[opt_idx[2]]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)  # Add outline for better visibility
            ),
            name='Optimal Design'
        )
        fig.add_trace(opt_point)
    
    # Update layout with fixed axes (same as before)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Power Split',
                range=[dim_bounds[0, 0], dim_bounds[0, 1]],
                autorange=False
            ),
            yaxis=dict(
                title='TOGA Throttle',
                range=[dim_bounds[1, 0], dim_bounds[1, 1]],
                autorange=False
            ),
            zaxis=dict(
                title='Cruise Throttle',
                range=[dim_bounds[2, 0], dim_bounds[2, 1]],
                autorange=False
            ),
            aspectmode='cube'  # Keep the axes proportional
        ),
        title=f"Design Space Visualization (Cost Weight: {weight:.2f})",
        margin=dict(l=0, r=60, b=10, t=40),  # Increased right margin for colorbar
        autosize=True
    )
    
    return fig, design_info_str

if __name__ == '__main__':
    print("Starting Dash server at http://127.0.0.1:8050/")
    app.run(debug=True)