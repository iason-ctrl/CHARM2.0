import numpy as np

from Plot_helpers import *
from CustomOperators import *
import os

ift.random.push_sseq_from_seed(19)

n_datapoints = 400
signal_space = ift.RGSpace(n_datapoints)
data_space = ift.UnstructuredDomain((n_datapoints,))

def radial_los(n_los):
    starts = [np.zeros(n_los)]
    ends = [np.array(ift.random.current_rng().uniform(0.1, 2.5, n_los))]
    return starts, ends

def random_mus(n_mus):
    return np.random.randint(100, size=n_mus)

# ------- Parameters of the correlated field model ------- #


# take the nomenclature from getting started 3 and such
# Alpha is not the fluctuations level
# fluctuations parameter doesn't really do anything because the sampled power spectra are not really bending in end_result.png !



'''
I don't expect the field to fluctuate much in the y-direction if at all. Maybe noise. 
- But overall, fluctations = (0.1, 1e-16)
- There is no reason to offset the field realizations, so set offset mean and offset std to None.
- I expect the slope to -4, I let it vary by maybe 1. loglogavgslope = (-4, 1)
- I expect no deviation from diagonal power law behaviour, so I set asperity to None and then as a consequency flexibility to None
'''


# I'm not sure what loglog avg slope std = -2 does differently from +2

args = {
    #"target_subdomain": signal_space,
    #"harmonic_partner": signal_space.get_default_codomain(),
    "offset_mean" :     None,
    "offset_std" :      None,
    "fluctuations":     (0.1, 1e-16),
    "loglogavgslope":   (-4,1),
    "asperity":         None,
    "flexibility":      None # use this
}
cf_info = str(list(args.values()))


# 'signal_cf' is the to be inferred signal field. I'm defining it with exp because my natural signal field is positive definite
signal_cf = ift.SimpleCorrelatedField(signal_space, **args)

signal = signal_cf

redshift_starts, redshift_ends =  radial_los(n_datapoints)
redshift_weights = (np.ones(n_datapoints)+redshift_ends)[0]


d_h = 3e8/(68.6e3)*1e6
noise = .001
N = ift.ScalingOperator(data_space, noise, np.float64)
HT = ift.HarmonicTransformOperator(signal_space.get_default_codomain(),signal_space)
R = ift.LOSResponse(signal_space, starts=redshift_starts, ends=redshift_ends)
REDSHIFTS_in = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,redshift_weights))
REDSHIFTS_out = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,redshift_weights))
FIVES = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,5*np.ones(n_datapoints)))

signal_response = 5*ift.log(d_h*REDSHIFTS_out(R(REDSHIFTS_in(ift.exp(-1/2*signal)))))
print("SIgnal response", signal_response)
#signal_response = integral(ift.exp(-1/2*signal))-ift.DiagonalOperator(ift.Field(domain=ift.DomainTuple.make(signal_space),val=10*np.ones(n_datapoints)))

# Generate mock signal and data
mock_position = ift.from_random(signal_response.domain, 'normal')

data_realization = signal_response(mock_position) + N.draw_sample()

#custom_plot(signal(mock_position).val,False, True, reconstruct=actual_signal_cf(mock_position).val)

#print("FOR PLOTTING: signal(mock_pos)" ,signal(mock_position)," R.adjoint times data realiz." ,R.adjoint_times(data_realization), "power spec list ",len(signal_cf.power_spectrum.force(mock_position).val))



# Plot Setup
plot = ift.Plot()
plot.add(signal(mock_position), title="Ground truth")
plot.add(R.adjoint_times(data_realization), title="Data realizations")
plot.add([signal_cf.power_spectrum.force(mock_position)], title="Power Spectrum")
path_groundTruths = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/"
plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths +'_'+f"{cf_info}_dataRealization.png")

# Minimization and sampling  controllers
ic_sampling_lin = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=0.05, iteration_limit=100)
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Sampling (nonlinear)", deltaE=0.5, iteration_limit=15, convergence_level=2)
ic_newton_minimization = ift.AbsDeltaEnergyController(name="Newton Minimization. Searching for energy descent direction", deltaE=0.5, iteration_limit=35, convergence_level=2)

minimizer_geoVI_MGVI = ift.NewtonCG(ic_newton_minimization)
nonlinear_geoVI_minimizer = ift.NewtonCG(ic_sampling_nl)

likelihood_energy = ift.GaussianEnergy(data_realization, inverse_covariance=N.inverse) @ signal_response

global_iterations = 6
n_samples = lambda iiter: 10 if iiter < 5 else 50 # increase sample rate for higher initial indices (get first a ball park and then converge with force)
samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                          total_iterations=global_iterations,
                          n_samples=n_samples,
                          kl_minimizer=minimizer_geoVI_MGVI,
                          sampling_iteration_controller=ic_sampling_lin,
                          nonlinear_sampling_minimizer= nonlinear_geoVI_minimizer,
                          output_directory="/Users/iason/PycharmProjects/Nifty/CHARM")

plot = ift.Plot()
mean, var = samples.sample_stat(signal)
plot.add(mean, title="Posterior Mean")
plot.add(var.sqrt(), title="Posterior Standard Deviation")

nsamples = samples.n_samples
logspectrum = signal_cf.power_spectrum.log()
plot.add(list(samples.iterator(signal_cf.power_spectrum)) +
         [signal_cf.power_spectrum.force(mock_position), samples.average(logspectrum).exp()],
         title="Sampled Posterior Power Spectrum",
         linewidth=[1.]*nsamples+[3,3],
         label=[None]*nsamples + ["Ground Truth", "Posterior mean"])

#print("FOR CUSTOM ROUTINE: mean",mean, " and std.dev ->",var.sqrt())
path_endresults = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/"
plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths +'_'+f"{cf_info}_endResult.png")

custom_plot(R.adjoint_times(data_realization).val,True,False,reconstruct=[],cf_info=cf_info)
custom_plot(signal(mock_position).val,False, True, reconstruct=mean.val,cf_info=cf_info)

print("saved result as end_result.png")
