import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerLine2D,HandlerErrorbar

palette = sns.color_palette('muted')
blue, orange, green, red, mauve, brown, pink, gray, yellow, lightblue = palette

def planck_cosmology(x):
    # x is natural redshift units
    return np.log(0.314*np.exp(3*x)+0.686)

def natural_x_coords_from_redshift(z):
    return np.log(1+z)

def initial_data_plot(x,y,x_natural):
    plt.subplot(2, 2, 1)

    plt.plot(x, y, ".", color="black")
    plt.title(r"Distance moduli $\mu$ against CMB corrected redshifts $z$")
    plt.xlabel(r"Redshift $z$")
    plt.ylabel(r"Distance modulus $\mu$")

    plt.subplot(2, 2, 2)

    plt.plot(x_natural, y, ".", color="black")
    plt.title(r"Distance moduli $\mu$ against CMB corrected redshifts $z$")
    plt.xlabel(r"Natural coordinates $x=ln(1+z)$")
    plt.ylabel(r"Distance modulus $\mu$")

    plt.subplot(2, 2, 3)

    plt.plot(x_natural, planck_cosmology(x_natural), "--", color="black")
    plt.title(r"Planck Cosmology: Evolution of cosmic energy density in time")
    plt.xlabel(r"Natural coordinates $x=ln(1+z)$ (measure of time)")
    plt.ylabel(r"Signal Field $s(x)$")

    plt.show()



def initial_data_plot_only_standard_diagram(x,y,x_natural):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 15}

    matplotlib.rc('font', **font)

    sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
    palette = sns.color_palette('deep')


    plt.plot(x_natural, y, ".", color=palette[0])
    plt.title(r"Distance moduli $\mu$ against CMB corrected redshifts $z$")
    plt.xlabel(r"Natural coordinates $x=ln(1+z)$")
    plt.ylabel(r"Distance modulus $\mu$")


    plt.show()


def plot_histogramms(redshifts,moduli):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 15}

    matplotlib.rc('font', **font)

    sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
    palette = sns.color_palette('deep')

    plt.subplot(2,1,1)
    plt.hist(redshifts,bins=1701,ec=palette[0],color=palette[0])
    plt.xlabel("Redshift $z$")
    plt.ylabel("Frequency")
    plt.title("Histogram of redshift frequency in the data")

    plt.subplot(2, 1,2)
    plt.xlabel("Distance modulus $\mu$")
    plt.ylabel("Frequency")
    plt.hist(moduli,bins=1701,ec=palette[3],color=palette[3])
    plt.title("Histogram of distance modulus frequency in the data")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def custom_plot(x,mode,name,abszisse=None,reconstruct=None, secondYaxis=None, secondXaxis=None, thirdYAxis=None,realData=None,
                PlanckData = None, ReconstructedData = None, deviation=False):
    n_datapoints = 0
    try:
        n_datapoints = len(realData)
    except:
        print("Number of datapoints could not have been calculated, input is not array or None.")

    plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"
    plt.rcParams["font.size"] = "12"


    fig, ax = plt.subplots()

    ax.tick_params(axis='both',direction='in',width=1.5)
    ax.tick_params(which='major',direction='in', length=7, )
    ax.tick_params(which='minor',direction='in', length=4, )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if mode=="DrawData":
        plt.plot(abszisse,x, markersize=8, marker="o", color="black", markerfacecolor='white',linewidth=0,label=f"datapoints ({n_datapoints})")
        #plt.plot(abszisse,x, "-", color="black", linewidth=2)
        plt.xlabel("Redshifts $z$")
        plt.ylabel(r"Distance modulus $\mu$")
        plt.title(r"Real data: Distance moduli $\mu$ against CMB corrected redshifts $z$", size=16)
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" +name+".png",dpi=300,bbox_inches='tight')
    elif mode=="SyntheticReconstruction":
        plt.plot(abszisse,reconstruct, color=blue, linewidth=1.5, ls="--", label="Signal Reconstruction")
        plt.plot(abszisse,x, color="black",linewidth=1, label="Ground truth")
        plt.xlabel("Natural redshift coordinates $x=log(1+z)$")
        plt.ylabel(r"Signal field $s(x):=log(\rho/\rho_0)\propto \rho$")
        plt.title("Mock signal and Reconstruction", size=16)
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" + name+".png",dpi=300,bbox_inches='tight')
    elif mode=="GroundTruth":
        plt.plot(abszisse, x, "-", color="black", linewidth=2, label="Signal")
        plt.xlabel("Natural redshift coordinates $x=log(1+z)$")
        plt.ylabel(r"Exponential of signal field $s(x):=log(\rho/\rho_0)\propto \rho$")
        plt.title("Ground truth", size=16)
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" + name + ".png", dpi=300, bbox_inches='tight')
    elif mode=="Histogramm":

        plt.plot(x, np.ones(len(x)), markersize=8, marker="o", color="black", markerfacecolor='white', linewidth=0,label="Datapoints")

        mean = np.mean(x)
        error = mean/2
        std = np.std(x)

        plt.axvline(mean,color=blue,label=f"Mean of distribution: {np.round(mean,2)}")
        plt.errorbar(mean,1,yerr=0,xerr=error,color=red,label=f"Standard deviation: {np.round(std,2)}",linewidth=3,markeredgewidth=2,capsize=15,ls="",marker="o",ecolor=red,)

        plt.legend(handles=[plt.plot([], marker="o", color="black", markerfacecolor="white",linewidth=0)[0],
                            plt.plot([], ls="-", color=blue)[0],
                            plt.plot([], ls="-", color=red)[0]],
                   labels=[f"Datapoints ({n_datapoints})",
                           f"Mean of distribution: {np.round(mean,2)}",
                           f"Standard deviation: {np.round(std,2)}"
                           ])
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.xlabel("Redshift range $z$")
        plt.ylabel("")
        plt.title("1D Distribution of redshifts of synthetic data")
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/"+name+".png",dpi=300, bbox_inches='tight')

    elif mode=="RealReconstruction":
        if deviation:
            plt.subplot(2, 2, 1)
        plt.errorbar(abszisse, x, yerr=thirdYAxis,color=blue,ecolor=lightblue,elinewidth=1.5, linewidth=3,ls="-", label="Reconstruction mean (Lightblue: standard deviation)")
        #plt.plot(np.linspace(0,5,300),np.cos(planck_cosmology(np.linspace(0,5,300))),color=red,linewidth=2,ls="-",label="Cosine of planck cosmology")
        plt.plot(np.linspace(0, 5, 300), planck_cosmology(np.linspace(0, 5, 300)), color=red, linewidth=2,ls="--", label="Planck cosmology")
        plt.plot(np.log(np.ones(n_datapoints)+secondXaxis), realData, marker="o", markerfacecolor="white", color="black", lw=0, label="Real moduli")
        plt.plot(np.log(np.ones(n_datapoints) + secondXaxis), ReconstructedData, marker="o", markerfacecolor="white",color=orange, lw=0, label="Reconstructed moduli")
        plt.plot(np.log(np.ones(n_datapoints) + secondXaxis), PlanckData, marker="o", markerfacecolor="white",color=blue, lw=0, label="Planck Moduli")
        plt.xlabel("Natural redshift coordinates $x=log(1+z)$")
        plt.ylabel(r"Signal field $s(x):=log(\rho/\rho_0)\propto \rho$")
        plt.title("Result of reconstruction", size=16)
        plt.legend()

        if deviation:
            plt.subplot(2, 1, 2)
            plt.plot(abszisse, np.zeros(len(abszisse)), color="black", linewidth=1, ls="--",label="Null line")
            plt.plot(abszisse, x-planck_cosmology(abszisse), color=mauve, linewidth=1,ls="-", label="Deviation")
            plt.xlabel("Natural redshift coordinates $x=log(1+z)$")
            plt.ylabel(r"$\Delta s(x)$")
            plt.title("Deviation of reconstruction from planck cosmology", size=16)

            plt.subplots_adjust(hspace=0.4)

        plt.show()
        #fig.set_size_inches(16, 9)
        #plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" + name+".png",dpi=300,bbox_inches='tight')

        """plt.clf()
        plt.plot(secondXaxis, secondYaxis, color=orange, label="Data produced by posterior",markersize=8, marker="o", markerfacecolor='white', linewidth=0,)
        plt.plot(secondXaxis, thirdYAxis, color=blue, label="Data produced by planck curve", markersize=8, marker="o",
                 markerfacecolor='white', linewidth=0, )
        plt.plot(secondXaxis, realData, color="black", label="Real data", markersize=8, marker="o",
                 markerfacecolor='white', linewidth=0, )
        plt.xlabel("Redshifts $z$")
        plt.ylabel(r"Distance modulus $\mu$")
        plt.title("Data resulting from the posterior mean, planck cosmology and real data")
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" + name + "dataRealization.png", dpi=300, bbox_inches='tight')

"""

def visualize_power_spectrum_parameters():


    plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"
    plt.rcParams["font.size"] = "12"

    def power_spectrum(k,alpha, beta):
        return beta*k**alpha

    x=np.linspace(0.01,10,300)

    fig, ax = plt.subplots()

    plt.text(0.55, 0.9, r"$\alpha$ $\in$ $[-6,-0.5]$. Lighter shades correspond to bigger $\alpha$'s",
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax.transAxes)

    ax.tick_params(axis='both',direction='in',width=1.5)
    ax.tick_params(which='major',direction='in', length=7, )
    ax.tick_params(which='minor',direction='in', length=4, )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    color = matplotlib.cm.get_cmap("gray")
    print("color",color)
    #for alpha, beta in zip(np.linspace(-6,-0.5,8),np.linspace(-10,10,8)):
    for alpha in np.linspace(-6,-0.5,8):
        plt.plot(x,power_spectrum(x,alpha,1), "-", color="black", linewidth=2, alpha=abs(alpha/-6))
    plt.xlabel("wavevectors $k$ in log scale")
    plt.ylabel("power_spec$(k)$ in log scale")
    plt.title(r"Influence of exponent on a power spectrum of the form $k^{\alpha}$ on log-log scale", size=16)
    #plt.xlim(-2,10)
    #plt.ylim(-2, 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.set_size_inches(16, 9)
    plt.savefig('/Users/iason/Downloads/Influence of exponent on power spectrum on log log scale.png', dpi=300,bbox_inches='tight')


