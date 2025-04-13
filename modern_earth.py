from POSEIDON.core import create_star, create_planet, define_model, wl_grid_constant_R,\
    make_atmosphere, read_opacities, compute_spectrum
from POSEIDON.constants import R_E, M_E, R_Sun
from POSEIDON.utility import read_PT_file, read_chem_file, plot_collection
from POSEIDON.visuals import plot_spectra
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

#***** Wavelength grid *****#

wl_min = 0.4    # Minimum wavelength (um)
wl_max = 2.0     # Maximum wavelength (um)

R = 10000         # Spectral resolution of grid

wl = wl_grid_constant_R(wl_min, wl_max, R)

#***** Define stellar properties *****#

R_s = 1.0*R_Sun   # Stellar radius (m)
T_s = 5772          # Stellar effective temperature (K)
Met_s = 0.0        # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.4        # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, wl=wl)

#***** Define planet properties *****#

planet_name = 'Modern Earth'  # Planet name used for plots, output files etc.

R_p = 1.0*R_E     # Planetary radius (m)
M_p = 1.0*M_E       # Planetary mass (kg)
T_eq = 255.0           # Equilibrium temperature (K)
a_p = (1*u.au).to(u.m).value

# Create the planet object
planet = create_planet(planet_name, R_p, mass = M_p, T_eq = T_eq, a_p=a_p)

# Define the model

model_name = 'Modern_Earth_Reflection'

# Atmosphere species

bulk_species = ['N2']
param_species = ['O', 'O2', 'H2O', 'OH', 'CO', 'CH4', 'CH3', 'C2H6', 'NO', 'NO2', \
                'H2S', 'SO', 'SO2', 'OCS', 'O3', 'N2O']


param_species = ['O2', 'CO2', 'CH4', 'H2O', 'O3', 'N2O']

# Define the model

surfaces = {
'ocean': 0.7156642408629538,
'engelmann': 0.008921276007386613,
'magnolia-grandiflora': 0.026687325186179115,
'aspen': 0.008890315511056323,
'adenostoma-fasciculatum': 0.008079888628937042,
'red-brown-loam': 0.018853073467519764,
'acacia-visco': 0.0223767501695908,
'rangeland': 0.09131223204470754,
'wetland': 0.002964508778645415,
'wheat-residue': 0.025945397032264846,
'asphalt': 0.0016609995422282827,
'antarctic': 0.02872956113530409,
'sand': 0.03991443163322637
}

surface_components = list(surfaces.keys())
print(surface_components)

model = define_model(model_name, bulk_species, param_species, 
                       PT_profile = 'file_read', X_profile = 'file_read', 
                       radius_unit = 'R_E', surface = True,  # <----- Set surface = True
                       reflection = True, #scattering = True, # <----- Set reflection and scattering to True
                       surface_model = 'lab_data',
                       cloud_model = 'cloud-free',           # <----- Set surface_model to 'lab_data'
                       surface_components = surface_components) # <----- Input surface_components


# Specify the pressure grid of the atmosphere
P_min = 1.0e-7    # 0.1 ubar
P_max = 10.0      # 10 bar (you can extend the atmosphere deeper than the surface)
N_layers = 100    # 100 layers

P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify location of the P-T profile file
PT_file_dir = 'data/PTprofile'
PT_file_name = 'PTprofile-Sun-1.000-FGKM_SRFALB_Earth0.5630_Cloud0.4370-1.0bar.txt'

# Specify location of the chemical equilibrium file
chem_file_dir = 'data/CHEM_profiles'
chem_file_name = 'CHEM-Sun-Earth_subset.txt'

# Read the P-T profile files
T_Modern = read_PT_file(PT_file_dir, PT_file_name, P, skiprows = 1,
                        P_column = 2, T_column = 3)

chem_in_file = ['O', 'O2', 'H2O', 'OH', 'CO', 'CH4', 'CH3', 'C2H6', 'NO', 'NO2', \
                'H2S', 'SO', 'SO2', 'OCS', 'O3', 'N2O']

X_Modern = read_chem_file(chem_file_dir, chem_file_name, P, chem_in_file, 
                          chem_species_in_model = model['chemical_species'], 
                          skiprows = 1)

# Make the atmosphere model

P_ref = 1      # We'll set the reference pressure at the surface
R_p_ref = R_p  # Radius at reference pressure

log_P_surf = 0        # Surface pressure is 1 bar
# Purple_sulfur_bacteria_pink_berries_E53_mud_dry_percentage = 0.5 # 50% purple cyanobacteria (dried up)
# Lunar_mare_basalt_percentage = 0.5 # 50% moon rock

surface_params = np.array([
    log_P_surf, 
    surfaces['ocean'],
    surfaces['engelmann'],
    surfaces['magnolia-grandiflora'],
    surfaces['aspen'],
    surfaces['adenostoma-fasciculatum'],
    surfaces['red-brown-loam'],
    surfaces['acacia-visco'],
    surfaces['rangeland'],
    surfaces['wetland'],
    surfaces['wheat-residue'],
    surfaces['asphalt'],
    surfaces['antarctic'],
    surfaces['sand'],
    ])

# surface_params = np.array([
#     log_P_surf, 
#     # Purple_sulfur_bacteria_pink_berries_E53_mud_dry_percentage,
#     # Lunar_mare_basalt_percentage
#     0.3
#     ]) #<---- Put surface params into new list, surface_params

# Generate the atmosphere
atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                T_input = T_Modern, X_input = X_Modern,
                                surface_params = surface_params)   #<---- Put surface params into make_atmosphere   

#***** Read opacity data *****#

opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 100     # 100 K lower limit covers the TRAPPIST-1e P-T profile
T_fine_max = 300     # 300 K upper limit covers the TRAPPIST-1e P-T profile
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 0.0    # 1 bar is the surface pressure, so no need to go deeper
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step)

# Create opacity object (both models share the same molecules, so we only need one)
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine, opacity_database = 'Temperate')

# import ipdb; ipdb.set_trace()
# Compute the spectrum
spectrum_FpFs, albedo = compute_spectrum(planet, star, model, atmosphere, opac, wl, \
                                    spectrum_type = 'emission', return_albedo = True, use_photosphere_radius= True)

spectra = []   # Empty plot collection

# Add the three model spectra to the plot collection object
spectra = plot_collection(spectrum_FpFs, wl, collection = spectra)

fig_spec = plot_spectra(spectra, planet)

fig_spec.savefig('./figs/Exo-Earth_Model.png', dpi = 300)

import ipdb
ipdb.set_trace()