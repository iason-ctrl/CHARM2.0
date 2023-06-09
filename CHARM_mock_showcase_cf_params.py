from Plot_helpers import *
from CustomOperators import *
import os

ift.random.push_sseq_from_seed(19)

n_datapoints = 200
signal_space = ift.RGSpace(n_datapoints)
data_space = ift.UnstructuredDomain((n_datapoints,))

def radial_los(n_los):
    starts = [np.zeros(n_los)]
    ends = list(ift.random.current_rng().random((n_los, 1)).T)
    return starts, ends

def random_mus(n_mus):
    return np.random.randint(100, size=n_mus)


def main(parameter, hint, clock):
    # ------- Parameters of the correlated field model ------- #

    alpha = 1 # AMPLITUDE OF FLUCTUATIONS / SPECTRAL ENERGY
    alpha_std = 1.4 # STANDARD DEVIATION FROM ALPHA VALUE
    k_exponent = -3 # SLOPE OF POWER SPECTRUM
    k_exponent_std = -1 # Let it wiggle a bit about . .

    # take the nomenclature from getting started 3 and such
    # Alpha is not the fluctuations level
    # fluctuations parameter doesnt really do anything because the sampled power spectra are not really bending in end_result.png !
    # redshift range should be included in my data

    # Where do the distance moduli come into play ?

    offset_mean = 2
    offset_std = (4, 0.8)

    args = {
        #"target_subdomain": signal_space,
        #"harmonic_partner": signal_space.get_default_codomain(),
        "offset_mean" :     offset_mean,
        "offset_std" :      offset_std,
        "fluctuations":     (alpha,alpha_std),
        "loglogavgslope":   (k_exponent,k_exponent_std),
        "asperity":         (1,0.1),
        "flexibility":      (parameter,0.005) # use this
    }
    cf_info = str(list(args.values()))


    # 'signal_cf' is the to be inferred signal field. I'm defining it with exp because my natural signal field is positive definite
    signal_cf = ift.SimpleCorrelatedField(signal_space, **args)

    signal = signal_cf

    redshift_stars, redshift_ends =  radial_los(n_datapoints)
    redshift_weights = np.exp(np.random.random(40))

    print("Redshift starts and ends", redshift_stars, redshift_ends)

    noise = .001
    N = ift.ScalingOperator(data_space, noise, np.float64)
    R = ift.LOSResponse(signal_space, starts=redshift_stars, ends=redshift_ends)
    signal_response = ift.log(R(ift.exp(-1/2*signal)))
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
    path_groundTruths = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/GroundTruths/"
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_groundTruths + f"_{hint}_" + f"{str(clock)+'_'+cf_info}.png")

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
    path_endresults = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/EndResults/"
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_endresults + f"_{hint}_" + f"{str(clock)+'_'+cf_info}.png")

    custom_plot(R.adjoint_times(data_realization).val,True,False,reconstruct=[],cf_info=cf_info,clock=clock,hint=hint)
    custom_plot(signal(mock_position).val,False, True, reconstruct=mean.val,cf_info=cf_info,clock=clock, hint=hint)

    print("saved result as end_result.png")


list_to_vary = np.linspace(0.1,4,9)
clock = 0

for loglogavgslope in list_to_vary:
    clock += 1
    main(loglogavgslope,hint="flexibility_mean",clock=clock)
    os.system('say "Eine weitere Iteration fertig."')

os.system('say "Script ist ausgeführt worden"')