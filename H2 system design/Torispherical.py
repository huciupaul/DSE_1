import numpy as np

V_init = 5.986
L_init = 3.44
R_init = 0.8018
t2 = 0.001

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

print(f"Volume of tank: {2 * V_tor + V_cyl} m^3")
print(f"Volume of torispherical head: {V_tor} m^3")
print(f"Total length of tank: {L_cyl + 2 * h} m")
print(f"Head length: {h} m")