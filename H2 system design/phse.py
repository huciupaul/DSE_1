import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

fluid = 'parahydrogen'

# Constants
T_triple = 13.8  # K
P_triple = 0.07703  # bar
T_crit = PropsSI('Tcrit', fluid)         # ~32.94 K
P_crit = PropsSI('Pcrit', fluid) / 1e5   # bar

# Temperature range
T_vals = np.linspace(T_triple, 50, 400)

# Saturation curve (liquid-vapor boundary)
T_sat = []
P_sat = []
for T in T_vals:
    try:
        P = PropsSI('P', 'T', T, 'Q', 0, fluid) / 1e5  # bar
        T_sat.append(T)
        P_sat.append(P)
    except:
        continue

# Plotting
plt.figure(figsize=(10, 6))
plt.yticks(np.arange(0, 21, 1))
plt.xticks(np.arange(0, 51, 2))
# Saturation curve
plt.plot(T_sat, P_sat, 'navy', lw=2, label='Saturation Curve')

# Critical point
plt.plot(T_crit, P_crit, 'o', color='darkgreen', label='Critical Point')
plt.axvline(T_crit, color='green', linestyle=':', label=f'{T_crit:.2f} K')
plt.axhline(P_crit, color='green', linestyle=':', label=f'{P_crit:.3f} bar')

# Region labels
plt.text(18.5, 7.15, 'Liquid Phase', fontsize=10)
plt.text(35, 7.15, 'Gaseous Phase', fontsize=10)
plt.text(35, 14, 'Supercritical\nPhase', fontsize=10, color='gray')
plt.text(18.5, 14, 'Compressible\nLiquid', fontsize=10, color='gray')

# Axis settings
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
plt.title('Parahydrogen Phase Diagram')
plt.xlim(T_triple + 0.5, 40)
plt.ylim(0.1, 20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
