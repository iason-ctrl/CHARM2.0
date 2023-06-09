import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

file = open("numpy_values.txt")
array = [float(x.strip()) for x in file.readlines()]
print(array)

def power(k,alpha):
    return k**(alpha)


x=np.linspace(1,21,21)
popt, = sc.optimize.curve_fit(power,x,array)


print(*popt)
plt.plot(array,"b.")
plt.plot(x,power(x,*popt),"b-")
plt.show()