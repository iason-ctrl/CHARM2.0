from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import numpy as np

c = 299792458
n0 = 1.42
alpha = [48.25, 44.55, 45.50, 45.05, 44.55, 43.90, 44.05, 43.25, 43.15, 42.95, 42.90]
alphakorr = [54.0, 48.25, 46.55, 45.50, 45.05, 44.55, 43.90, 43.55, 43.25, 43.15, 42.95, 42.90]
lambd = np.arange(500 * 1e-9, 1001 * 1e-9, 50 * 1e-9)
lambdkorr = np.arange(450 * 1e-9, 1001 * 1e-9, 50 * 1e-9)


def alpha_in_k(x, lam):
    return (2 * np.pi) * n0 / (lam) * np.sin(2 * np.pi * x / 360)


def lambda_in_w(h):
    return 2 * np.pi * c / h


xLichtgerade = np.linspace(0.5e7, 2.5 * 1e7, 10000)


# xLichtgerade = [0,1.6*1e7]
def Lichtgerade(k, n0):
    return k * c / n0


yLichtgerade = []
for i in xLichtgerade:
    yLichtgerade.append(Lichtgerade(i, n0))

y2Lichtgerade = []  # Lichtgerade Luft
for m in xLichtgerade:
    y2Lichtgerade.append(Lichtgerade(m, 1))

kPlasmon = []
for j in range(len(alpha)):
    kPlasmon.append(alpha_in_k(alpha[j], lambd[j]))

wPlasmon = []
for l in lambd:
    wPlasmon.append(lambda_in_w(l))

kPlasmonkorr = []
for n in range(len(alphakorr)):
    kPlasmonkorr.append(alpha_in_k(alphakorr[n], lambdkorr[n]))

wPlasmonkorr = []
for o in lambdkorr:
    wPlasmonkorr.append(lambda_in_w(o))
plt.plot(xLichtgerade, y2Lichtgerade, 'b', linewidth=0.75)
plt.plot(xLichtgerade, yLichtgerade, 'orange', linewidth=0.75)
plt.plot(kPlasmonkorr, wPlasmonkorr, 'r+', linewidth=0.5)
plt.plot(kPlasmon, wPlasmon, 'k+', linewidth=0.5)

plt.xlabel('Wellenvektor ' r'$k$ in 1/m')
plt.ylabel('Frequenz' r'$\ \omega$ in 1/s')
plt.legend(['Lichtgerade in Luft', 'Lichtgerade im Prisma', 'korrigierte Messwerte', 'Messwerte'])
plt.show()

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import numpy as np

c = 299792458
n0 = 1.42
e2 = 8.8541878128 * 1e-12
alpha = [48.25, 44.55, 45.50, 45.05, 44.55, 43.90, 44.05, 43.25, 43.15, 42.95, 42.90]
alphakorr = [54.0, 48.25, 46.55, 45.50, 45.05, 44.55, 43.90, 43.55, 43.25, 43.15, 42.95, 42.90]
lambd = np.arange(500 * 1e-9, 1001 * 1e-9, 50 * 1e-9)
lambdkorr = np.arange(450 * 1e-9, 1001 * 1e-9, 50 * 1e-9)


def alpha_in_k(x, lam):
    return (2 * np.pi) * n0 / (lam) * np.sin(2 * np.pi * x / 360)


def lambda_in_w(h):
    return 2 * np.pi * c / h


xLichtgerade = np.linspace(0.5e7, 2.5 * 1e7, 10000)


# xLichtgerade = [0,1.6*1e7]
def Lichtgerade(k, n0):
    return k * c / n0


def f(x, A, B, C, D):
    return x / c * ((A / ((x + B) ** C) + D) * e2 / (A / ((x + B) ** C) + D + e2)) ** 0.5

def func(x,A,B,C,D):
    E = A/(x+B)**C + D
    eps = 1/0.6+1/E
    res = x * eps**(-1/2)
    return x*(eps**(-1/2))


def Fit(x, y):
    params, covariance_matrix = curve_fit(f, x, y, bounds=([0, 0, 0, -1e6], [1e10, 1e10, 1e5, 1e16]))
    errors = np.sqrt(np.diag(covariance_matrix))
    # print('A =', params[0], '±', errors[0])
    # print('B =', params[1], '±', errors[1])
    # print('C =', params[2], '±', errors[2])
    return params


yLichtgerade = []
for i in xLichtgerade:
    yLichtgerade.append(Lichtgerade(i, n0))

y2Lichtgerade = []  # Lichtgerade Luft
for m in xLichtgerade:
    y2Lichtgerade.append(Lichtgerade(m, 1))

kPlasmon = []
for j in range(len(alpha)):
    kPlasmon.append(alpha_in_k(alpha[j], lambd[j]))

wPlasmon = []
for l in lambd:
    wPlasmon.append(lambda_in_w(l))

kPlasmonkorr = []
for n in range(len(alphakorr)):
    kPlasmonkorr.append(alpha_in_k(alphakorr[n], lambdkorr[n]))

wPlasmonkorr = []
for o in lambdkorr:
    wPlasmonkorr.append(lambda_in_w(o))

#Fitten = Fit(kPlasmonkorr, wPlasmonkorr)
#print(Fitten)
yFit = []

xarr=np.linspace(0,max(xLichtgerade),1000)
plt.plot(func(np.linspace(-50,50,300),4,5,6,5),"g-")
plt.show()

popt, _= curve_fit(f,kPlasmonkorr, wPlasmonkorr,bounds=([0, 0, 0, -1e6], [1e10, 1e10, 1e5, 1e16]))
print(popt)

for p in xLichtgerade:
    yFit.append(f(i, Fitten[0], Fitten[1], Fitten[2], Fitten[3]))
plt.plot(xLichtgerade, y2Lichtgerade, 'b', linewidth=0.75)
plt.plot(xLichtgerade, yLichtgerade, 'orange', linewidth=0.75)
plt.plot(kPlasmonkorr, wPlasmonkorr, 'r+', linewidth=0.5)
plt.plot(kPlasmon, wPlasmon, 'k+', linewidth=0.5)
#plt.plot(xLichtgerade, yFit,"r-")

plt.plot(xarr, f(xarr,*popt),"r-")
plt.xlabel('Wellenvektor ' r'$k$ in 1/m')
plt.ylabel('Frequenz' r'$\ \omega$ in 1/s')
plt.legend(['Lichtgerade in Luft', 'Lichtgerade im Prisma', 'korrigierte Messwerte', 'Messwerte'])
plt.show()