import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trivialHelpers import *
from CustomOperators import *
plt.rcParams['figure.dpi'] = 100


'''
We shall use a Correlated Field to make parts of the covariance spectrum which we do not know part of the inference problem.
In our case we know the exponent of the power law (-4 = avg log log slope with standard deviation = 0), but not the 
spectral strength / amplitude, which I hope is my alpha parameter = fluctutations. 
I'm relatively sure you can't make t_bg (i.e. your prior / likelihood energy) part of the inference problem,
through a CF-Model at least. 
'''

# ------- Read Data ------- #

required_fields = ["zCMB","MU_SH0ES"]

df = pd.read_csv("../Pantheon+SH0ES.csv", sep=" ", skipinitialspace=True, usecols=required_fields)
x = df.zCMB
y = df.MU_SH0ES
# Natalia calls this the 'x coordinates in data space'
x_natural = np.array([natural_x_coords_from_redshift(z) for z in x])
x_linspace = np.linspace(0,max(x_natural),len(x_natural))

'''
We denote samples from the set of 'natural' x coordinates x_natural as x_j
'''

#initial_data_plot_only_standard_diagram(x,y,x_natural)
#initial_data_plot(x,y,x_natural) # Initial Plot of Data and Planck Cosmology


# ------- Define signal space and harmonic counter space, data space and field ------- #

data_space = ift.UnstructuredDomain((len(df.MU_SH0ES),))
data_field = ift.Field(ift.DomainTuple.make(data_space),val = df.MU_SH0ES)
signal_space = ift.RGSpace(len(df.zCMB)) # 'x-space domain'
harmonic_signal_space = signal_space.get_default_codomain()

redshift_starts = np.array([np.zeros(len(x_natural))])
redshift_ends = np.array([x_natural])


HT = ift.HarmonicTransformOperator(harmonic_signal_space,signal_space)
Sh = ift.create_power_operator(harmonic_signal_space,pow_spec,sampling_dtype=float)
#R = HT

sh = Sh.draw_sample()

redshift_field = ift.Field(ift.DomainTuple.make(signal_space),val=x_natural)

R = ift.LOSResponse(signal_space,redshift_starts,redshift_ends)
R2 = WeightedLOSResponse(domain=signal_space,starts=redshift_starts,ends=redshift_ends,integration_weight=redshift_field.val)
signal_samples = HT(sh)

noiseless_data = R(signal_samples)
noiseless_data_structured_domain = ift.RGSpace((1701,))
noiseless_data_structured= ift.Field(ift.DomainTuple.make(noiseless_data_structured_domain),val=noiseless_data.val)

noise_amplitude = np.sqrt(10)
N = ift.ScalingOperator(signal_space,noise_amplitude**2,float)
n = ift.Field.from_random(domain=ift.DomainTuple.make(signal_space),random_type='normal',std=noise_amplitude, mean=2)


d = noiseless_data_structured + n
print(" data",d.val)

s_data = HT(sh).val
d_data = d.val
print(min(d_data),max(d_data))
plt.ylim(min(d_data), max(d_data))
plt.plot(s_data, 'r', label="Signal", linewidth=2)
plt.plot(s_data, 'r.',)
#plt.yscale(value="log")
#plt.xscale(value="log")
#plt.plot(np.linspace(0,1701,6000),np.linspace(0,1701,6000)**(-4),"-",color="orange",label="k^-4 function with scaled y axis")
#plt.plot(sh.val,".",label="harmonic variance values sample", color="gray")
plt.title("Reconstruction")
plt.plot(d_data, 'k.', label="Data")
plt.legend()
plt.show()

redshift_field = ift.Field(ift.DomainTuple.make(signal_space),val=x_natural)
x_values = np.linspace(0,np.log(1+max(df.zCMB)),signal_space.size) # Equidistant points in the range 0 - max signal

# ------- Parameters of the correlated field model ------- #

alpha = 2 # AMPLTIUDE OF FLUCTUATIONS / SPECTRAL ENERGY
alpha_std = 0.1 # STANDARD DEVIATION FROM ALPHA VALUE
k_exponent = -4 # SLOPE OF POWER SPECTRUM
k_exponent_std = 0 # THEORETICALLY NO DEVIATION FROM k^-4 SPECTRUM

offset_mean = 2
offset_std = (1e-3, 1e-6)

args = {
    "target_subdomain": signal_space,
    "harmonic_partner": harmonic_signal_space,
    "fluctuations":     (alpha,alpha_std),
    "loglogavgslope":   (k_exponent,k_exponent_std),
    "asperity":         (0.001,0.0005),
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

exponentiated_and_scaled_cf =np.exp(-0.5*signal_cf)
los = WeightedLOSResponse(signal_space,redshift_starts,redshift_ends,integration_weight=redshift_field.val)


#manipulated_cf = exponentiated_and_scaled_cf*np.exp(x_natural)

signal_response = los(exponentiated_and_scaled_cf)

d_H = 1.315*10**26

'''
Iteration control, set up of energy model and choice of minimizer (<- geoVI) for the Kullback-Leibler-Divergence.
Via minimization of the KLD in the geoVI algorithm, we get an approximated Posterior.
'''


# non-linear iteration control and minimizer
ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',deltaE=0.5, iteration_limit=30,convergence_level=3)
ic_newton = ift.AbsDeltaEnergyController(name="Newton", deltaE=0.01, iteration_limit=35)


noise = .001 # TODO this from my dataset probably. Maybe noise is a function of data_space so different for each point in data space not constant
N = ift.ScalingOperator(data_space, noise, float)

# Gaussian Energy does (Rs-d) instead of what I expected -(d-Rs) but thats ofc ok
likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response
energy = ift.StandardHamiltonian(likelihood_energy,ic_sampling_nl)

ic_sampling_nl.enable_logging()
ic_newton.enable_logging()

minimizer = ift.NewtonCG(ic_newton,enable_logging=True)

# the difference between geoVI and mgvi is the expansion point of SampledKLEnergy: geoVI finds the optimal point
# for linear approximation of the coordinate transformation
# you start with a random sample from the hamiltonian

# Where does SampledKLEnergy know when to use geoVI and when to use MGVI ?

geoVI_init_pos = ift.from_random(energy.domain, 'normal')

posterior_samples = ift.SampledKLEnergy(hamiltonian=energy,position=geoVI_init_pos,n_samples=50,minimizer_sampling=minimizer,mirror_samples=False)

mean_unsliced, var_unsliced = posterior_samples.samples.sample_stat(exponentiated_and_scaled_cf)

plot = ift.Plot()

plot.add(
        ift.Field.from_raw(signal_space, mean_unsliced.val),
        title="Posterior Unsliced Mean",
    )

plot.output(xsize=15, ysize=15, name="maahah1.png".format("results"))
print("Saved results as", "maahah1.png".format("results"))