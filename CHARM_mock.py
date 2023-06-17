import numpy as np

# TODO NIFTY BUG : IFT.LOG(IFT.EXP()) = () WITH THE SAME DOMAIN

from Plot_helpers import *
from CustomOperators import *
import os

ift.random.push_sseq_from_seed(19)

x_length = 7
n_datapoints = 200
n_pix = 400
signal_space = ift.RGSpace(n_pix,distances=x_length/n_pix)
data_space = ift.UnstructuredDomain((n_datapoints,))

signal_data_space = ift.UnstructuredDomain((n_pix,))

def radial_los(n_los):
    starts = [np.zeros(n_los)]
    ends = [np.array(abs(ift.random.current_rng().normal(0.05,0.4,n_los)))]
    return starts, ends


# ------- Parameters of the correlated field model ------- #


# take the nomenclature from getting started 3 and such
# Alpha is not the fluctuations level
# fluctuations parameter doesn't really do anything because the sampled power spectra are not really bending in end_result.png !



'''
I don't expect the field to fluctuate much in the y-direction if at all. Maybe noise. 
- But overall, fluctuations = (0.1, 1e-16)
- There is no reason to offset the field realizations, so set offset mean and offset std to None.
- I expect the slope to -4, I let it vary by maybe 1. loglogavgslope = (-4, 1)
- I expect no deviation from diagonal power law behaviour, so I set asperity to None and then as a consequency flexibility to None
'''


# I'm not sure what loglog avg slope std = -2 does differently from +2

args = {
    #"target_subdomain": signal_space,
    #"harmonic_partner": signal_space.get_default_codomain(),
    "offset_mean" :     None, # this has an effect on data realizations (y-Verschiebung)
    "offset_std" :      None,
    "fluctuations":     (0.1, 1e-16), # this 'bends' the data to a curve at high redshifts
    "loglogavgslope":   (4,-1), # this has no effect on data realizations
    "asperity":         None, # this has no effect on data realizations
    "flexibility":      None # this has no effect on data realizations
}
cf_info = str(list(args.values()))


# Ensure positivity of the to be inferred field via exponentiating.
# Take the logarithm later because nifty interprets ift.log(ift.exp(signal)) = signal without changing the domain
signal_cf = ift.SimpleCorrelatedField(signal_space, **args)
signal = signal_cf

signal_coordinate_field = Unity(signal_space)

redshift_starts, redshift_ends =  radial_los(n_datapoints)
redshift_weights = np.ones(n_datapoints)+redshift_ends[0]
natural_redshifts = np.log(np.ones(n_datapoints)+redshift_ends[0])

# NATURAL REDSHIFTS HAVE TO BE WHAT IS INTEGRATED OVER . SO HERE , x = redshift_ends[0] and z = e^x - 1


d_h = 3e8/(68.6e3)*1e6
noise = .1
N = ift.ScalingOperator(data_space, noise, np.float64)
R = ift.LOSResponse(signal_space, starts=redshift_starts, ends=redshift_ends)
#REDSHIFTS_in = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,redshift_weights))
REDSHIFTS_out = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,np.exp(redshift_ends[0])))
SUBTRACT_FIVES = ift.Adder(a=-5,domain=ift.DomainTuple.make(data_space,))


redshifty_weights_structured = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,np.exp(signal_coordinate_field.val)))
#redshifty_weights_unstructured = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_data_space,np.exp(signal_coordinate_field.val)))

"""one = redshifty_weights_structured(ift.exp(-1/2*signal))
two = R(one)
three = REDSHIFTS_out(two)
four = SUBTRACT_FIVES(5*np.log10(d_h*three))"""


signal_response = SUBTRACT_FIVES(5*np.log10(d_h*REDSHIFTS_out(R(redshifty_weights_structured(ift.exp(-1/2*signal))))))


# Generate mock signal and data ensure positivity of synthetic ground truth.
mock_position = ift.from_random(signal_response.domain, 'normal', mean=10, std=1 )
data_realization = signal_response(mock_position) + N.draw_sample()

unity = UnityOfDomain(ift.DomainTuple.make(signal_space,))



signal_coordinate_field = Unity(signal_space)

custom_plot(data_realization.val,mode="DrawData", name="Synthetic_data_realization",abszisse=np.exp(redshift_ends[0])-np.ones(n_datapoints))

custom_plot(signal(mock_position).val,mode="GroundTruth",name="TEST NORMAL Synthetic_Data_GroundTruth",abszisse=signal_coordinate_field.val)
custom_plot(signal_coordinate_field.val,mode="GroundTruth",name="TEST NORMAL UNITY OPERATOR Synthetic_Data_GroundTruth",abszisse=signal_coordinate_field.val)


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


path_endresults = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/"
plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths +'_'+f"{cf_info}_endResult.png")

custom_plot(signal(mock_position).val,mode="SyntheticReconstruction",reconstruct=mean.val,name="synthetic_signal_reconstruction",abszisse=signal_coordinate_field.val)

print("saved result as end_result.png")
