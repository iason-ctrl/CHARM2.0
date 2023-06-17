import numpy as np
import pandas as pd
from Plot_helpers import *
from CustomOperators import *
import scipy as sc

def planck_cosmology(x):
    return np.log(0.314*np.exp(3*x)+0.686)
d_h = 3e8/(68.6e3)*1e6

# ------- Read Data ------- #


required_fields = ["zCMB","MU_SH0ES"]

df = pd.read_csv("Pantheon+SH0ES.csv", sep=" ", skipinitialspace=True,usecols=required_fields)
redshifts = np.array(df.zCMB)
moduli = np.array(df.MU_SH0ES)

n_pix = 8000
x_length = 6.7 # 6.7 = natural redshift of CMB ~ 'beginning' of the universe. length != 6* max(x_natural) = 6*1.2  distances != x_length/n_pix  so range is 2.1=x_natural_max

n_datapoints = len(redshifts)

redshift_weights = 1 + redshifts
natural_redshifts = np.log(np.ones(1701)+redshifts)

integrand = redshift_weights*np.exp(-1/2*planck_cosmology(np.linspace(min(natural_redshifts),max(natural_redshifts),1701)))
integral = []
for z in redshifts:
    integral_val = sc.integrate.simpson(y=integrand,x=np.linspace(0,np.log(1+z),len(integrand)))
    integral.append(integral_val)
integral = np.array(integral)
data_produced_by_planck = 5*np.log10(d_h*integral*redshift_weights)-5

plt.plot(redshifts,moduli,"b.")
plt.plot(np.linspace(min(natural_redshifts),max(natural_redshifts),1701),data_produced_by_planck,"r.")
#plt.show()





print("Number of datapoints: " , n_datapoints)
redshift_ends, redshift_starts = [np.log(np.ones(n_datapoints)+redshifts)], [np.zeros(n_datapoints)]


natural_redshift_range=np.linspace(min(natural_redshifts),max(natural_redshifts),n_datapoints)

signal_space = ift.RGSpace(n_pix,distances=x_length/n_pix)
signal_data_space = ift.UnstructuredDomain((n_pix,))

data_space = ift.UnstructuredDomain((n_datapoints,))
data_field = ift.Field(domain=ift.DomainTuple.make(data_space,),val=moduli)

# ------- Parameters of the correlated field model ------- #

custom_plot(moduli,mode="DrawData",abszisse=redshifts,name="Real_Moduli_vs_natural_redshifts")


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
    "offset_mean" :     3, # this has an effect on synthetic data realizations (y-Verschiebung)
    "offset_std" :      None,
    "fluctuations":     (1,1e-16), # this was (3, 1) this 'bends' the synthetic data realizations to a curve at high redshifts
    "loglogavgslope":   (-4,1e-16), # this was (-4, 1) this has no effect on synthetic data realizations
    "asperity":         None, # this has no effect on synthetic data realizations
    "flexibility":      None # this has no effect on synthetic data realizations
}
cf_info = str(list(args.values()))

signal_cf = ift.SimpleCorrelatedField(signal_space, **args)
signal = signal_cf


redshift_weights = np.ones(n_datapoints)+redshift_ends[0]
natural_redshifts = np.log(np.ones(n_datapoints)+redshift_ends[0])

signal_coordinate_field = Unity(signal_space)


d_h = 3e8/(68.6e3)*1e6
noise = .1
N = ift.ScalingOperator(data_space, noise, np.float64)
R = ift.LOSResponse(signal_space, starts=[np.zeros(n_datapoints)], ends=[np.log(np.ones(n_datapoints)+redshifts)])
REDSHIFTS_in = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,redshift_weights))

REDSHIFTS_out = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,np.ones(n_datapoints)+redshifts))
SUBTRACT_FIVES = ift.Adder(a=-5,domain=ift.DomainTuple.make(data_space,))
unity = UnityOfDomain(ift.DomainTuple.make(signal_space))


redshifty_weights_structured = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,np.exp(signal_coordinate_field.val)))
redshifty_weights_unstructured = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_data_space,np.exp(signal_coordinate_field.val)))


one = redshifty_weights_structured(ift.exp(-1/2*signal))
two = R(one)
print("step two ", two.domain)
three = REDSHIFTS_out(two)
four = SUBTRACT_FIVES(5*np.log10(d_h*three))

signal_response = SUBTRACT_FIVES(5*np.log10(d_h*(REDSHIFTS_out(R(redshifty_weights_structured(ift.exp(-1/2*signal)))))))

# Minimization and sampling  controllers
ic_sampling_lin = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=0.05, iteration_limit=100)
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Sampling (nonlinear)", deltaE=0.5, iteration_limit=15, convergence_level=2)
ic_newton_minimization = ift.AbsDeltaEnergyController(name="Newton Minimization. Searching for energy descent direction", deltaE=0.5, iteration_limit=35, convergence_level=2)

minimizer_geoVI_MGVI = ift.NewtonCG(ic_newton_minimization)
nonlinear_geoVI_minimizer = ift.NewtonCG(ic_sampling_nl)

likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response

global_iterations = 6
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

"""
nsamples = samples.n_samples
logspectrum = signal.power_spectrum.log()
plot.add(list(samples.iterator(signal.power_spectrum)) +
         [samples.average(logspectrum).exp()],
         title="Sampled Posterior Power Spectrum",
         linewidth=[1.]*nsamples+[3,3],
         label=[None]*nsamples + ["Ground Truth", "Posterior mean"])

path_endresults = "/Users/iason/PycharmProjects/Nifty/CHARM/figures/"
plot.output(ny=1, nx=3, xsize=24, ysize=6, name=path_endresults +'_'+f"{cf_info}_endResult.png")"""


integrand = redshift_weights*np.exp(-1/2*planck_cosmology(np.linspace(min(natural_redshifts),max(natural_redshifts),n_datapoints)))
integral = []
for z in redshifts:
    integral_val = sc.integrate.simpson(y=integrand,x=np.linspace(0,np.log(1+z),len(integrand)))
    integral.append(integral_val)
integral = np.array(integral)

data_produced_by_planck = 5*np.log10(d_h*integral*redshift_weights)-5

integrand = redshift_weights*np.exp(-1/2*mean.val[0:1701])
integral = []
for z in redshifts:
    integral_val = sc.integrate.simpson(y=integrand,x=np.linspace(0,np.log(1+z),len(integrand)))
    integral.append(integral_val)
integral = np.array(integral)

data_produced_by_posterior = 5*np.log10(d_h*integral*redshift_weights)-5


custom_plot(x=mean.val,mode="RealReconstruction",name="CHARM2_0_Result",abszisse=signal_coordinate_field.val,
            secondYaxis = natural_redshift_range, secondXaxis = redshifts, realData = moduli, thirdYAxis=var.sqrt().val,
            ReconstructedData=data_produced_by_posterior, PlanckData=data_produced_by_planck, deviation=False)

print("saved result as CHARM2_0_Result.png")
os.system('say "Plot Routinen Ende."')