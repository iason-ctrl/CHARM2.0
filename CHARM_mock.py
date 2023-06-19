
from Plot_helpers import *
from CustomOperators import *

# ------- Read Data for comparison ------- #

redshifts_real, moduli_real = read_data(keyword="Pantheon+") # 1701 datapoints.
redshift, mu, Noise_natalia = read_data(keyword="Union2.1") # for comparison. 580 datapoints. Noise is a 580x580 matrix


# ------- Some constants ------- #

d_h = 3e8/(68.6e3)*1e6                   # Hubble-length in parsecs
noise = 0.001                            # Noise level original Charm: ~ 0.0007

# ------- Basic parameters and function------- #

ift.random.push_sseq_from_seed(19)
n_pix = 400
x_length = 6.7 # 6.7 = natural redshift of CMB ~ 'beginning' of the universe.
n_datapoints = 200
print("Number of datapoints: " , n_datapoints)

def radial_los(n_los):
    starts = [np.zeros(n_los)]
    ends = [np.array(abs(ift.random.current_rng().normal(0.05,0.4,n_los)))]
    return starts, ends

# ------- Define domains and basic fields. Save a plot of the data. ------- #

redshift_starts, redshift_ends = radial_los(n_datapoints)
redshifts = np.exp(redshift_ends[0]) - 1

signal_space = ift.RGSpace(n_pix,distances=x_length/n_pix)
signal_data_space = ift.UnstructuredDomain((n_datapoints,))

data_space = ift.UnstructuredDomain((n_datapoints,))



# ------- Build the correlated field model ------- #

'''
I don't expect the field to fluctuate much in the y-direction if at all. Maybe noise. 
- But overall, fluctations = (0.1, 1e-16)
- There is no reason to offset the field realizations, so set offset mean and offset std to None.
- I expect the slope to -4, I let it vary by maybe 1. loglogavgslope = (-4, 1)
- I expect no deviation from diagonal power law behaviour, so I set asperity to None and then as a consequency flexibility to None
'''


# I'm not sure what loglog avg slope std = -2 does differently from +2

args = {
    "offset_mean" :     None, # this has an effect on synthetic data realizations (y-Verschiebung)
    "offset_std" :      None,
    "fluctuations":     (1,1e-16), # this was (3, 1) this 'bends' the synthetic data realizations to a curve at high redshifts
    "loglogavgslope":   (-4,1e-16), # this was (-4, 1) this has no effect on synthetic data realizations
    "asperity":         None, # this has no effect on synthetic data realizations
    "flexibility":      None # this has no effect on synthetic data realizations
}
cf_info = str(list(args.values()))

signal_cf = ift.SimpleCorrelatedField(signal_space, **args)
signal = signal_cf

# ------- Build necessary fields and operators for signal response ------- #

signal_coordinate_field = Unity(signal_space)               # a 1D array containing the argument of the correlated field i.e. x in s = s(x)

weights_data_space = np.ones(n_datapoints) + redshifts      # corresponds to 1 + z = e^x. There n_datapoints many 1+z values.
weights_signal_space = np.exp(signal_coordinate_field.val)  # corresponds to e^x. There n_pix many e^x values.

WEIGHT_signal_space = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,weights_signal_space))  # Weight operator in signal space
WEIGHT_data_space = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,weights_data_space))        # Weight operator in data space

SUBTRACT_FIVES = ift.Adder(a=-5,domain=ift.DomainTuple.make(data_space,))

N = ift.ScalingOperator(data_space, noise, np.float64)
R = ift.LOSResponse(signal_space, starts=redshift_starts, ends=redshift_ends)


# ------- Build the signal response as chain of operators  ------- #


# Build the response operator as a chain of operators step by step to detect any errors
control_step_one = WEIGHT_signal_space(ift.exp(-1/2*signal))
control_step_two = R(control_step_one)
control_step_three = WEIGHT_data_space(control_step_two)
control_step_four = SUBTRACT_FIVES(5*np.log10(d_h*control_step_three))

# Build the response operator chain in one line
signal_response = SUBTRACT_FIVES(5*np.log10(d_h*(WEIGHT_data_space(R(WEIGHT_signal_space(ift.exp(-1/2*signal)))))))

# ------- Synthetic data  ------- #

mock_position = ift.from_random(signal_response.domain, 'normal', mean=10, std=1 )
data_realization = signal_response(mock_position) + N.draw_sample()
data_field = ift.Field(domain=ift.DomainTuple.make(data_space,),val=data_realization.val)

custom_plot(x=[],realData=data_realization.val,show=False,save=False,mode="DrawData",abszisse=redshifts,name="Synthetic_Moduli_VS_Redshifts")

# Plot Setup
plot = ift.Plot()
plot.add(signal(mock_position), title="Ground truth")
plot.add(R.adjoint_times(data_realization), title="Data realizations")
plot.add([signal_cf.power_spectrum.force(mock_position)], title="Power Spectrum")
path_groundTruths = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/"
plot_ift_plot = False
if plot_ift_plot:
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths +'_'+f"{cf_info}_dataRealization.png")

# ------- Minimization and sampling  controllers ------- #


ic_sampling_lin = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=0.05, iteration_limit=100)
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Sampling (nonlinear)", deltaE=0.01, iteration_limit=60, convergence_level=2)
ic_newton_minimization = ift.AbsDeltaEnergyController(name="Newton Minimization. Searching for energy descent direction", deltaE=0.5, iteration_limit=35, convergence_level=2)

minimizer_geoVI_MGVI = ift.NewtonCG(ic_newton_minimization)
nonlinear_geoVI_minimizer = ift.NewtonCG(ic_sampling_nl)

likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response

global_iterations = 15
n_samples = lambda iiter: 10 if iiter < 5 else 50 # increase sample rate for higher initial indices (get first a ball park and then converge with force)
samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                          total_iterations=global_iterations,
                          n_samples=n_samples,
                          kl_minimizer=minimizer_geoVI_MGVI,
                          sampling_iteration_controller=ic_sampling_lin,
                          nonlinear_sampling_minimizer= nonlinear_geoVI_minimizer,
                          output_directory="/Users/iason/PycharmProjects/Nifty/CHARM")

os.system('say "Kullback Leibler Divergenz wurde minimiert, Plot Routinen fangen an."')

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
if plot_ift_plot:
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths +'_'+f"{cf_info}_endResult.png")

data_produced_by_posterior, data_produced_by_planck = create_data_from_signal(redshifts=redshifts,
                                                                              mean_values=mean.val,
                                                                              planck=True, n_datapoints=n_datapoints)



custom_plot(save=False,show=True,ground_truth=signal(mock_position).val,x=mean.val,mode="SyntheticReconstruction",name="Synthetic_Reconstruction",abszisse=signal_coordinate_field.val,
            secondXaxis = redshifts, realData = moduli_real,thirdXaxis = np.log(np.ones(1701)+redshifts_real), thirdYAxis=var.sqrt().val,
            ReconstructedData=data_produced_by_posterior,
            deviation=True, PlanckData=data_produced_by_planck,
            secondYaxis=data_realization.val)

print("saved result as CHARM2_0_Result.png")
os.system('say "Plot Routinen Ende."')