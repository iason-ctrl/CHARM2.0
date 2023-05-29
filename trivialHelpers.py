import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

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