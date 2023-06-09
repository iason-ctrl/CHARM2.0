import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trivialHelpers import *
from CustomOperators import *

'''
USE MOCK !! draw samples 
'''


# ------- Read Data ------- #

required_fields = ["zCMB","MU_SH0ES"]

df = pd.read_csv("Pantheon+SH0ES.csv", sep=" ", skipinitialspace=True,usecols=required_fields)
x = df.zCMB
y = df.MU_SH0ES

x_natural = np.array([natural_x_coords_from_redshift(z) for z in x])


#initial_data_plot_only_standard_diagram(x,y,x_natural)
#initial_data_plot(x,y,x_natural) # Initial Plot of Data and Planck Cosmology
plot_histogramms(x,y)

# ------- Define signal space and harmonic counter space, data space and field ------- #

data_space = ift.UnstructuredDomain((len(df.MU_SH0ES),))
data_field = ift.Field(ift.DomainTuple.make(data_space),val = df.MU_SH0ES)
signal_space = ift.RGSpace(len(df.zCMB)) # 'x-space domain'
harmonic_signal_space = signal_space.get_default_codomain()

redshift_field = ift.Field(ift.DomainTuple.make(signal_space),val=x_natural)


# ------- Parameters of the correlated field model ------- #

alpha = 2 # AMPLTIUDE OF FLUCTUATIONS / SPECTRAL ENERGY
alpha_std = 1.4 # STANDARD DEVIATION FROM ALPHA VALUE
k_exponent = -4 # SLOPE OF POWER SPECTRUM
k_exponent_std = 1 # Let it wiggle a bit about . .

offset_mean = 2
offset_std = (1.2, 0.8)

args = {
    "target_subdomain": signal_space,
    "harmonic_partner": harmonic_signal_space,
    "fluctuations":     (alpha,alpha_std),
    "loglogavgslope":   (k_exponent,k_exponent_std),
    "asperity":         None,
    "flexibility":      (0.001,0.0005) # use this
}

# 'signal_cf' is the to be inferred signal field
cf_creator = ift.CorrelatedFieldMaker(prefix="harmonic")
cf_creator.add_fluctuations(**args)
cf_creator.set_amplitude_total_offset(offset_mean=offset_mean,offset_std=offset_std)

signal_cf = cf_creator.finalize()



# the correlated field is defined over a bunch of domains (consists of a bunch of fields) but their target points to signal space...
# so cf.target = signal space
# we want our operator to go from signal space to data space R : signals space -> data space || Rs /-> mu
redshift_starts = np.array([np.zeros(len(x_natural))])
redshift_ends = np.array([x_natural])

customLOSresponse = WeightedLOSResponse(signal_space,redshift_starts,redshift_ends,integration_weight=redshift_field.val)
signal_response = customLOSresponse(signal_cf)
d_H = 1.315*10**26

'''
Iteration control, set up of energy model and choice of minimizer (<- geoVI) for the Kullback-Leibler-Divergence.
Via minimization of the KLD in the geoVI algorithm, we get an approximated Posterior.
'''


# non-linear iteration control and minimizer
ic_energy = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',deltaE=0.5, iteration_limit=30,convergence_level=3)
ic_energy.enable_logging()

minimizer = ift.NewtonCG(ic_energy,enable_logging=True)

noise = .001 # TODO this from my dataset probably. Maybe noise is a function of data_space so different for each point in data space not constant
N = ift.ScalingOperator(data_space, noise, float)

# Gaussian Energy does (Rs-d) instead of what I expected -(d-Rs) but thats ofc ok
likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response
energy = ift.StandardHamiltonian(likelihood_energy,ic_energy)

# the difference between geoVI and mgvi is the expansion point of SampledKLEnergy: geoVI finds the optimal point
# for linear approximation of the coordinate transformation
# you start with a random sample from the hamiltonian

# Where does SampledKLEnergy know when to use geoVI and when to use MGVI ?

geoVI_init_pos = ift.from_random(energy.domain, 'normal')

posterior_samples = ift.SampledKLEnergy(hamiltonian=energy,position=geoVI_init_pos,n_samples=50,minimizer_sampling=minimizer,mirror_samples=False)

mean_unsliced, var_unsliced = posterior_samples.samples.sample_stat(signal_cf)

plot = ift.Plot()

plot.add(
        ift.Field.from_raw(signal_space, mean_unsliced.val),
        title="Posterior Unsliced Mean",
    )

plot.output(xsize=15, ysize=15, name="maahah1.png".format("results"))
print("Saved results as", "maahah1.png".format("results"))