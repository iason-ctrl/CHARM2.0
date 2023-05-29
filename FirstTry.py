import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trivialHelpers import *
from CustomFunctions import *

'''
We shall use a Correlated Field to make parts of the covariance spectrum which we do not know part of the inference problem.
In our case we know the exponent of the power law (-4 = avg log log slope with standard deviation = 0), but not the 
spectral strength / amplitude, which I hope is my alpha parameter = fluctutations. 
I'm relatively sure you can't make t_bg (i.e. your prior / likelihood energy) part of the inference problem,
through a CF-Model at least. 
'''

# ------- Read Data ------- #

required_fields = ["zCMB","MU_SH0ES"]

df = pd.read_csv("Pantheon+SH0ES.csv", sep=" ", skipinitialspace=True,usecols=required_fields)
x = df.zCMB
y = df.MU_SH0ES
# Natalia calls this the 'x coordinates in data space'
x_natural = np.array([natural_x_coords_from_redshift(z) for z in x])
x_linspace = np.linspace(0,max(x_natural),len(x_natural))
'''
We denote samples from the set of 'natural' x coordinates x_natural as x_j
'''

#initial_data_plot_only_standard_diagram(x,y,x_natural)
initial_data_plot(x,y,x_natural) # Initial Plot of Data and Planck Cosmology


# ------- Define signal space and harmonic counter space, data space and field ------- #

signal_space = ift.RGSpace(len(df.zCMB)) # 'x-space'
harmonic_signal_space = signal_space.get_default_codomain()
HarmonicTransformator = ift.HarmonicTransformOperator(domain=harmonic_signal_space, target=signal_space)

data_space = ift.UnstructuredDomain((len(df.MU_SH0ES),))
data_field = ift.Field(ift.DomainTuple.make(data_space),val = df.MU_SH0ES)

""" signal response needs to be a field ... okay. You take the signal field (i.e. our
corr. field) and you only measure points from that field. Number of points = Number of redshifts.
Then you fold each redshift (x_j=f(z)) with the corresponding corr field domain point = g(x_j/z) or something?
But the thing is, the correlated field is a chain operator. 
"""

# ------- Define Background Cosmology fields t_bg ------- #
x_values = np.linspace(0,np.log(1+max(df.zCMB)),signal_space.size) # Equidistant points in the range 0 - max signal
t_bg_agnostic = ift.Field(domain=ift.DomainTuple.make(signal_space), val = 2 * x_values) # TODO: here, val should be a hyperparameter to be optimized for (thus making t_bg a hyperparameter)
t_bg_planck =  ift.Field(domain=ift.DomainTuple.make(signal_space), val = planck_cosmology(x_values))

# ------- Define Operators ------- #

power_space = ift.PowerSpace(harmonic_signal_space) # 1D spectral space on which the power spectrum is defined
#power_spectrum_field = ift.Field(ift.DomainTuple.make(harmonic_signal_space), val= lambda k: power_spectrum(k))


Variance_harmonic = ift.create_power_operator(domain=harmonic_signal_space,power_spectrum=power_spectrum, sampling_dtype=float)
harmonic_signal_samples = Variance_harmonic.draw_sample()


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
    "flexibility":      (0.001,0.0005)
}

# 'signal_cf' is the to be inferred signal field
cf_creator = ift.CorrelatedFieldMaker(prefix="harmonic")
cf_creator.add_fluctuations(**args)
cf_creator.set_amplitude_total_offset(offset_mean=offset_mean,offset_std=offset_std)

signal_cf = cf_creator.finalize()

def Responsö(corr_field):
    return 2

#print("HEHRR AreAResr ", [signal_cf(s) for s in ift.from_random(signal_cf.domain["target_subdomain"])])


SignalSlice = SingleDomain(domain=signal_cf.target,target=signal_space)
Response = SampleIntegrationOfCorrelatedFieldOperator(datadomain=data_space,x_natural=x_natural,signal_space_domain=signal_space,x_linspace=x_linspace,signal_space_as_cf=signal_cf)


# the correlated field is defined over a bunch of domains (consists of a bunch of fields) but their target points to signal space...
# so cf.target = signal space
# we want our operator to go from signal space to data space R : signals space -> data space || Rs /-> mu
TestOp = TestOperator(domain=signal_cf.target, target=data_space, x_natural_space=x_natural, x_natural_linspace= x_linspace)
signal_response = signal_cf.integrate()
#signal_response = response_operator(x_natural,SignalSlice)

#R = ift.FFTOperator(signal_space,harmonic_signal_space)
#signal_response = R(signal_cf)-ift.Field(ift.DomainTuple.make(signal_space),val= 5*np.ones(len(df.zCMB)))

'''
Iteration control, set up of energy model and choice of minimizer (<- geoVI) for the Kullback-Leibler-Divergence.
Via minimization of the KLD in the geoVI algorithm, we get an approximated Posterior.
'''


# non-linear iteration control and minimizer
ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',deltaE=0.5, iteration_limit=15,convergence_level=2)

noise = .001 # TODO this from my dataset probably. Maybe noise is a function of data_space so different for each point in data space not constant
N = ift.ScalingOperator(data_space, noise, float)

# Position von t_bg stimmt nicht aber Erinnerung das es halt dort hinzukommt
# Gaussian Energy does (Rs-d) instead of what I expected -(d-Rs)
# TODO Wo kommt t_background hin ?
likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response # TODO hier fehlt doch t_bg als Hyperparameter
energy = ift.StandardHamiltonian(likelihood_energy)

minimizer = ift.NewtonCG(ift.GradientNormController(iteration_limit=3, name='Mini'))

# the difference between geoVI and mgvi is the expansion point of SampledKLEnergy: geoVI finds the optimal point
# for linear approximation of the coordinate transformation
# you start with a random sample from the hamiltonian
geoVIpos = ift.from_random(energy.domain, 'normal')

posterior_samples = ift.SampledKLEnergy(hamiltonian=energy,position=geoVIpos,minimizer_sampling=minimizer,n_samples=20)

# TODO what we need to do here is invert s = R^{-1}(d-n) using unter anderem geoVI. Do I need the CF model? Why what does it mean?
# TODO I need it at least to specify the hyperparameters t_bg and \alpha
# TODO how th do I use geoVi