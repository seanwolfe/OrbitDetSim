import numpy as np

mu_e = 398600.1
m_s = 1.989E+30
G = 6.67430e-11 / (1000 ** 3)
mu_s = G * m_s
L = 149597870.691
T = np.sqrt(L ** 3 / (mu_e + mu_s))
print(f"{mu_e + mu_s:.5e}")
print(f"{T:.5e}")
print(f"{L/T:.5e}")