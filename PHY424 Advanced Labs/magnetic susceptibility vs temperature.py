# Import libraries
import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import curve_fit
from scipy import constants as cst



# Import constants
k_b = cst.Boltzmann
u_0 = cst.mu_0



# Define magnetic susceptibility as a function of temperature
def Tchi(x, iv, J, ns):
    return (2 * iv * u_0 / (k_b * x)) * (4 * np.e**(-J/(k_b * x)) / (1 + 3 * np.e**(-J/(k_b * x)) )) + ns



# Graph the general shape for a magnetic susceptibility vs temperature graph
TT = np.linspace(0.1, 2000.1, 1000)
XX = (1000 / TT) * (4 * np.e**(-300/TT) / (1 + 3 * np.e**(-300/TT) )) + 5.5
plt.figure()
ax = plt.gca()
ax.set_ylim([0, 16])
plt.axis('off')
plt.plot(TT, XX, color='black')
plt.axhline(y=5.0, color='black', linestyle='-')
plt.text(0, 4, 'low temperature')
plt.text(1450, 4, 'high temperature')
plt.savefig('shape.png')



# Load volatage data which is converted as temperature
T, V_emf = np.loadtxt('TV.txt', skiprows=1, delimiter=',', unpack=True)
l = len(V_emf)

T = T - 30
V_emf = (0.5168 - V_emf) * 1000



# Calculate uncertainty
V_emf_unc = np.zeros(l)
T_unc = np.zeros(l)
X = np.zeros(l)
X_unc = np.zeros(l)

cnst = 1 / (10**6 * (0.03 * 2000) * np.pi *0.00238**2 * u_0 * (2000) *  2 * np.pi * 1200 * 0.1)
for k in range(l):
    V_emf_unc[k] = 2
    T_unc[k] = 0.5
    X[k] = V_emf[k] * cnst
    X_unc[k] = V_emf_unc[k] * cnst
#see eq 7
T_arr = np.linspace(T[0], T[-1], 200)



# Plot V_emf (proportional to magnetic susceptibility) against T
plt.figure()
ax = plt.gca()
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.set_xlim([65, 280])
plt.errorbar(T, V_emf, xerr = T_unc, yerr = V_emf_unc, fmt='.', label='Data')
plt.xlabel("The temperature of the sample (K)")
plt.ylabel("The EMF induced in the secondary coils ($\mu$ V)")
plt.savefig('TV_plot.png')

plt.figure()
ax = plt.gca()
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.set_xlim([65, 280])
plt.errorbar(T, X, xerr = T_unc, yerr = X_unc, fmt='.', label='Data')
plt.xlabel("The temperature of the sample (K)")
plt.ylabel("The magnetic susceptibility of the sample (m^3/kg)")
plt.savefig('TX_plot.png')



# Fit and graph the data
popt, pcov = curve_fit(Tchi, T, X, sigma = X_unc, absolute_sigma = True, p0 = (1E-17, 3E-21, 0.03))
pstd = np.sqrt(np.diag(pcov))


X_fit = Tchi(T, popt[0], popt[1], popt[2])
X_arr_fit = Tchi(T_arr, popt[0], popt[1], popt[2])


plt.figure()
ax = plt.gca()
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.set_xlim([65, 280])
plt.errorbar(T, X, xerr = T_unc, yerr = X_unc, fmt='.', label='Data')
plt.plot(T_arr, X_arr_fit, color='green', label= 'Fit')
plt.xlabel("The temperature of the sample (K)")
plt.ylabel("The magnetic susceptibility of the sample (m^3/kg)")
plt.text(80, 0.07, 'reduced $\chi^2 = 154.9$')
plt.text(80, 0.08, '$J = (94.7 \pm 1.9)10^{-22}$  J')
plt.text(80, 0.09, '$\mu_B^2 = (230.0 \pm 9.8)10^{-18}$  H^2/m^2')
plt.savefig('TX_fit_plot.png')

# Residual Plot
plt.figure()
ax = plt.gca()
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
plt.errorbar(T, X - X_fit, yerr = X_unc, fmt='.', label='Gaussian Residual')
plt.axhline(y=0.0, color='black', linestyle='--')
plt.xlabel("The temperature of the sample (K)")
plt.ylabel("The magnetic susceptibility residue (m^3/kg)")
plt.savefig('TX_res.png')

dof = l - 3
chi2 = 1 / dof * sum( ( (X - X_fit) / X_unc )**2 )

# Print chi^2 value
print("chi2 is: ", chi2)
print(popt, pstd)
