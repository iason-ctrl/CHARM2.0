## Charm: cosmic history agnostic reconstruction method
##
## NIFTY applying iterative Wiener filter to reconstruct the cosmic 
## expansion history 
##
## Authors: Natalia Porqueres, Torsten Ensslin
##
## charm is licensed under the
##`GPLv3 <http://www.gnu.org/licenses/gpl.html>`
##--------------------------------------------------------------------------
from __future__ import division

from nifty import *
from scipy.integrate import simps
import matplotlib.pyplot as plt

execfile("parameter_file.py")


class full_response_operator(operator):
    #usage:
    #R = full_response_operator(signaldomain, datadomain, xsignal, xdata, background, initialpoint)

# signal response: signal space (cf ?) --> data space

    def __init__(self, domain, target, x, xd, t):
        self.domain = domain
        self.target = target
        self.x = x
        self.xd = xd # "x natural" x_signal natural form function of redshift
        self.t = t

        self.imp = True
        self.sym = False
        self.uni = False

        self.dh = 3e8/(68.6e3)*1e6   #pc**-1 

    def _multiply(self, phi):
        #define a field to save the integral at each position
        integral = field(self.target)
        #function of the integrand
        f = field(self.domain, val=exp(self.x-0.5*(self.t+phi)))
        #integral limits
        for j in range(0, len(integral)):
            #position of last pixel below upper limit of the integral
            ind = np.where(self.x < self.xd[j])[-1][-1]

            #check if the limit of the integral is in the x coordinates.
            if (abs(x[ind + 1]-xd[j])<1e-6):
                #include the point
                ind += 1
                integral[j] = exp(self.xd[j])*simps(f[0:ind],self.x[0:ind], dx=self.domain.vol[0])
                
            else:
                #extrapolate to the limit of the integral
                y2 = (self.xd[j]-self.x[ind])*(f[ind+1]
                                               -f[ind])/(self.x[ind+1]-self.x[ind]) + f[ind]
                
                integral[j] = exp(self.xd[j])*simps(np.append(f[0:ind], y2),
                             np.append(self.x[0:ind], self.xd[j]), dx=self.domain.vol[0])
        
        return 5.0*log(self.dh*integral, 10)-5.0



class lin_response(operator):
    """linear response operator

    usage:
    
    R = lin_response(signaldomain, datadomain, xsignal, xdata, background, initialpoint)
    """
    def __init__(self, xdomain, ddomain, x, xd, t):
        self.imp = True
        self.sym = False
        self.uni = False

        self.domain = xdomain
        self.target = ddomain

        self.x = x
        self.xd = xd
        self.t = t
        
        #define functions
        f = field(xdomain, val=exp(x-0.5*(t)))
        
        #define empty variables
        q = np.zeros([ddomain.dim(), xdomain.dim()])
        r0 = field(ddomain)

        #integral limits
        for j in range(0, len(r0)):
            
            #last pixel below upper limit
            ind = np.where(x < xd[j])[-1][-1]
            
            #step function
            theta = np.zeros(xdomain.dim())
            theta[0:ind] = 1

            #check if the x coordinates contains the upper limit
            if (abs(x[ind + 1]-xd[j])<1e-6):
                #add the pixel
                theta[ind+1] = 1
                q[j,:] = (f.val*theta)*exp(xd[j])
                r0[j] = exp(xd[j])*simps(f[0:ind+1], 
                                     x[0:ind+1], self.domain.vol[0])
            else:
                #extrapolate
                y2 = (xd[j]-x[ind])*(f[ind+1]-f[ind])/(x[ind+1] - x[ind]) + f[ind]  
                newf = f
                newf[ind+1] = y2
                theta[ind+1] = 1
                q[j,:] = (newf.val*theta)*exp(xd[j])
                r0[j] = exp(xd[j])*simps(np.append(f[0:ind+1],y2), 
                                     np.append(x[0:ind+1], xd[j]), self.domain.vol[0])          
            
        self.r0 = R.dh*r0
        self.q = R.dh*q
        
        #constant due to derivation
        self.a = 2.5/log(10)

    def _multiply(self, argument):

        #define output field
        output = field(self.target)

        #compute summations
        output = np.einsum("ij,i,j->i", self.q, 1/self.r0.val, argument.val)

        return (-self.a*output)

    def give_explicit(self):

        explicit_matrix = -self.a*np.einsum("ij, i->ij", self.q, 1/self.r0.val)

        return explicit_operator(self.domain, target=self.target, sym=False,
                                 uni=False, matrix=explicit_matrix, bare=True)

    def _adjoint_multiply(self, argument):

        output = field(self.domain)
        output = np.einsum("ij,i,i->j", self.q, 1/self.r0.val, argument.val)

        return (-self.a*output)/self.domain.vol[0]


class laplace(operator):
    """laplace operator

    usage:

    L = laplace(domain)
    """
    def __init__(self, domain):
        self.domain = domain
        self.target = domain

        self.imp = True
        self.uni = False
        self.sym = True
        self.para = None

    def _multiply(self, x):
        #compute derivatives
        ret = 2*x-np.append(0, x[:-1])-np.append(x[1:], 0)

        #correction in the boundaries
        ret[0] -= (2*x[0]-x[1])
        ret[-1] -= (x[-1])
        
        return -ret/(self.domain.vol[0])**2


class spectral_roughness(operator):
    """spectral roughness operator

    usage:

    T = spectral_roughness(domain)
    """
    def __init__(self, domain):
        self.domain = domain
        self.target = domain
        
        self.imp = True
        self.uni = False
        self.sym = True
        
        self.L = explicify(laplace(domain), loop=True)
        
    def _multiply(self, t):
        return self.L.transpose()(self.L(t))


def plot(xd,d,x,t,t0,D,S):
    #Planck cosmology to compare results
    tPlanck = field(x_space, val=log(0.314*exp(3*x)+0.686))
    
    #format
    plt.rc('text', usetex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    labelx = -0.09
    fig.subplots_adjust(hspace=0.01)

    #upper panel
    ax1.errorbar(x, t0, yerr=sqrt(S.diag(bare=False)), color='deepskyblue')
    ax1.errorbar(x, t, yerr=sqrt(D.diag(bare=True)), color='yellow', mec="yellow")
    ax2.set_ylim(0, 1.6)

    ax1.plot(x, t, color='black', label="reconstruction")
    ax1.plot(x, t0,color='red', label="background cosmology", linestyle="dotted")
    ax1.plot(x, tPlanck, color='blue', label="Planck cosmology", linestyle="dashed")

    ax1.set_ylabel("signal")
    ax1.yaxis.set_label_coords(labelx, 0.5)
    ax1.legend(bbox_to_anchor=(0.4,1), fontsize=11)

    
    #bottom panel: deviation from Planck
    ax2.set_ylim(-0.6, 0.9)
    
    ax2.errorbar(x, t0-tPlanck, yerr=sqrt(S.diag(bare=False)), color='deepskyblue')
    ax2.errorbar(x, t-tPlanck, yerr=sqrt(D.diag(bare=True)), color='yellow')
    ax2.plot(x, t0-tPlanck, color='red', ls="dotted", label="background cosmology")
    ax2.plot(x, t-t, color='black', linestyle="dashed")
    ax2.plot(x, t-tPlanck, color='black', label="reconstruction")
    
    ax2.set_xlabel("x = ln(1+z)")
    ax2.set_ylabel("deviation from \n Planck cosmology")

    plt.savefig('Reconstruction.png')
  
    
#-------Read data---------------------------------------------------------------

#open data file    
f = open(fdata, "r")

#read data and sort by redshift
z = np.loadtxt(fdata, usecols=[1], dtype=float)
index = np.argsort(z)
z = z[index]
mu = np.loadtxt(fdata, usecols=[2], dtype=float)[index]
errMu = np.loadtxt(fdata, usecols=[3], dtype=float)[index]
f.close()

#read covariance matrix
fCov = open(fCovSys, "r")
N = np.loadtxt(fCovSys, dtype=float)
fCov.close()

#---------Define space and coordinates----------------------------------------

#compute the x coordinates in data space
xd = log(np.ones(len(z))+z)

#define x coordinates in signal space
x = np.arange(start=0., stop=xd.max()+2*dist, step=dist)

#only even numbers of grid points are supported for nifty fields
if np.mod(len(x), 2)==1: 
    x = x[:-1]

#define signal space
x_space = rg_space(len(x), dist=dist, zerocenter=False)

#define data space
d_space = point_space(len(z))

#-------------Define fields-------------------------------------------------
#define agnostic background expansion model 
t = field(x_space, val=2*x)

#in case your prior is Planck cosmology, uncomment next line
#t = field(x_space, val=log(0.314*exp(3*x)+0.686))

#define data field
d = field(d_space, val=mu)

#-----------Define operators------------------------------------------------
#define prior
if verbose: print "computing S"
T = spectral_roughness(x_space)
Sinv = explicify(T)
S = Sinv.inverse()*x_space.vol[0]

#invert covariance matrix
if verbose: print "computing Ninv"
Ninv = np.linalg.inv(N)

#-------------Wiener filter----------------------------------------------
#save initial background expansion model
t0 = t

#set precission
tolerance = 1e-2

#to enter the loop
err = 1e10  

if verbose: print "Wiener Filter"

#iterate Wiener filter until fullfill tolerance
while err>tolerance:
    #define response
    R = full_response_operator(domain=x_space, target=d_space, x=x,
                                xd=xd, t=t)

    #define linear response
    Rlin = lin_response(xdomain=x_space, ddomain=d_space, x=x, xd=xd,
                        t=t)

    Rlin = Rlin.give_explicit()

    if verbose: print "j"

    #compute source function
    j = Rlin.adjoint_times(np.dot(Ninv, d-R(0.0))) - Sinv(t-t0)

    if verbose: print "Dinv"

    #compute propagator operator
    Dinv = Sinv + Rlin.adjoint()*Ninv*Rlin 

    if verbose: print "D"

    D = Dinv.inverse()
    
    if verbose: print "m = D(j)"

    #Wiener filter reconstruction
    m = D(j)

    #take maximum of the reconstructe perturbation as error
    err = max(abs(m))

    #update background expansion model
    t = t + m
    
plot(xd,d,x,t,t0,D,S)
if verbose: print "Reconstruction.png generated"
