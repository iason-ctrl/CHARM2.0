import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.patches import Rectangle
'''
import matplotlib.font_manager
fpaths = matplotlib.font_manager.findSystemFonts()

for i in fpaths:
    f = matplotlib.font_manager.get_font(i)
    print(f.family_name)
'''



def planck_cosmology(x):
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

def custom_plot(x,drawData,Reconstruction,reconstruct,cf_info):

    plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"
    plt.rcParams["font.size"] = "12"


    fig, ax = plt.subplots()

    ax.tick_params(axis='both',direction='in',width=1.5)
    ax.tick_params(which='major',direction='in', length=7, )
    ax.tick_params(which='minor',direction='in', length=4, )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if drawData==True:
        plt.plot(np.linspace(0,1,200),x, markersize=8, marker="o", color="black", markerfacecolor='white')
        plt.plot(np.linspace(0,1,200),x, "-", color="black", linewidth=2)
        plt.xlabel("signal space")
        plt.ylabel("data")
        plt.title("Data realization (mock)", size=16)
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" +f"{cf_info}_customDataRealization.png",dpi=300,bbox_inches='tight')
    elif Reconstruction==True:
        plt.plot(np.linspace(0,1,200),reconstruct, "-.", color="blue", linewidth=4, label="Reconstruction")
        plt.plot(np.linspace(0,1,200),x, "-", color="black", linewidth=2, label="Mock Signal")
        plt.xlabel("signal space")
        plt.ylabel("signal")
        plt.title("Mock signal and Reconstruction", size=16)
        plt.legend()
        fig.set_size_inches(16, 9)
        plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/" + f"{cf_info}_customSignalMockAndReconstruction.png",dpi=300,bbox_inches='tight')



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

visualize_power_spectrum_parameters()

