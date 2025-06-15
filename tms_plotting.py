import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from matplotlib.colors import to_rgba

import json


def fluids_table(fluids):
    # Filter fluids for hydrogen and coolant
    selected_fluids = fluids
    labels = []
    mass_flows = []
    pressures = []
    temperatures = []
    for key in fluids.keys():
        f = fluids[key]
        labels.append(key)
        mass_flows.append(f['massflow'])
        pressures.append(f['pressure'])
        temperatures.append(f['temperature'])
    # Convert pressure from Pascal to bar and round values to 4 significant figures
    mass_flows = [float(f"{v:.4g}") for v in mass_flows]
    pressures = [float(f"{v/1e5:.4g}") for v in pressures]  # 1 bar = 1e5 Pa
    temperatures = [float(f"{v:.4g}") for v in temperatures]
    df = pd.DataFrame({
        'Fluid': labels,
        'Mass Flow': mass_flows,
        'Pressure': pressures,
        'Temperature': temperatures
    })

    # Add a 'Type' column based on fluid name
    def fluid_type(name):
        if re.search(r'h2', name, re.IGNORECASE):
            return 'H2'
        elif re.search(r'cool', name, re.IGNORECASE):
            return 'Coolant'
        else:
            return 'Other'

    df['Type'] = df['Fluid'].apply(fluid_type)

    # Sort by Type then Fluid
    df = df.sort_values(['Type', 'Fluid']).reset_index(drop=True)

    # Prepare cell colors
    def get_colors(values):
        colors = []
        prev = None
        for v in values:
            if prev is None:
                colors.append('white')
            elif v > prev:
                colors.append('#b6fcb6')  # light green
            elif v < prev:
                colors.append('#fcb6b6')  # light red
            else:
                colors.append('#f0f0f0')  # gray
            prev = v
        return colors

    cell_colors = []
    for col in ['Mass Flow', 'Pressure', 'Temperature']:
        cell_colors.append(get_colors(df[col].values))

    # Add Fluid and Type columns as white
    cell_colors = [['white'] * len(df)] + cell_colors + [['white'] * len(df)]

    # Transpose to match table shape
    cell_colors = list(map(list, zip(*cell_colors)))

    # Plot table
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(df)))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellColours=cell_colors,
        loc='center'
    )
    # Make row height larger
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_height(0.04)  # Increase this value for larger rows
    table.auto_set_column_width(col=list(range(len(df.columns))))
    fig.suptitle('Fluid Properties Table', fontsize=14)
    plt.show()

def csv_fluids(fluids):

    # Convert fluids dictionary to DataFrame
    data = []
    for key, f in fluids.items():
        data.append({
            'Fluid': key,
            'Mass Flow (kg/s)': f['massflow'],
            'Pressure (bar)': f['pressure'] / 1e5,  # Convert Pa to bar
            'Temperature (K)': f['temperature']
        })
    df = pd.DataFrame(data)
    # Add a 'Type' column based on fluid name prefix
    def fluid_type(name):
        if name.startswith('H2_'):
            return 'H2'
        elif name.startswith('Cool_'):
            return 'Coolant'
        else:
            return 'Other'

    df['Type'] = df['Fluid'].apply(fluid_type)
    df = df.sort_values(['Type', 'Fluid']).reset_index(drop=True)
    # Save to CSV
    df.to_csv('fluids_properties.csv', index=False)
    print("Fluids properties saved to fluids_properties.csv")

def plot(pdf_path, fluids, positions, page_number=0, dpi=300):
    """
    Overlays fluid data onto a P&ID diagram.

    Args:
        pdf_path (str): Path to the PDF file.
        fluids (dict): Dictionary with keys like 'H2_1', 'Cool_3' etc. and values as
                       dicts with 'massflow', 'temperature', and 'pressure'.
        positions (dict): Dictionary mapping station names to (x, y) positions on the image.
        page_number (int): PDF page to extract (0-indexed).
        dpi (int): Resolution for rendering the PDF.
    """

    # Load PDF and render page as image
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(pix.width / dpi, pix.height / dpi), dpi=dpi)
    ax.imshow(pix.pil_image(), extent=(0, pix.width, pix.height, 0))  # Top-left origin

    # Overlay text at defined positions
    for station, data in fluids.items():
        if station in positions:
            x, y = positions[station]
            overlay_text = (
                f"{station}\n"
                f"{data['massflow']:.2f} kg/s\n"
                f"{data['pressure']:.1f} bar\n"
                f"{data['temperature']:.1f} °C"
            )
            ax.text(x, y, overlay_text, fontsize=3, color='blue' if 'H2' in station else 'green',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.axis('off')
    plt.tight_layout()
    plt.show()

def pick_positions_on_pid(pdf_path, dpi=300):

    # Load and render PDF page
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # first page
    pix = page.get_pixmap(dpi=dpi)
    
    fig, ax = plt.subplots(figsize=(pix.width / dpi, pix.height / dpi), dpi=dpi)
    ax.imshow(pix.pil_image(), extent=(0, pix.width, pix.height, 0))
    ax.set_title("Click to place stations. Close window when done.")

    positions = {}

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside axes
        x, y = event.xdata, event.ydata
        print(f"Clicked at: ({int(x)}, {int(y)})")
        name = input("Enter station name (e.g., H2_1 or Cool_2), or press Enter to skip: ").strip()
        if name:
            positions[name] = (int(x), int(y))
            ax.plot(x, y, 'ro')  # mark the clicked point
            ax.text(x + 5, y - 5, name, fontsize=8, color='red')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # Optionally save to JSON
    with open("station_positions.json", "w") as f:
        json.dump(positions, f, indent=4)
        print("Positions saved to station_positions.json")

    return positions