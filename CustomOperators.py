import os

import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt

# ------- Response Operator ------- #

# usage: signal_response = Response(correlated field)

class WeightedLOSResponse(ift.LOSResponse):
    def __init__(self, domain, starts, ends, integration_weight=None, sigmas=None, truncation=3.):
        super(WeightedLOSResponse,self).__init__(domain, starts, ends, sigmas, truncation=3.)
        self.weights = integration_weight



    def apply(self, x, mode):

        self._check_input(x, mode)

        exponentiated_weights = np.exp(self.weights)
        d_H = 1.315 * 10 ** 26

        '''
        We want to build R(s) = 5*log_{10} (e^x * d_H * int[0,x]  e^{-0.5 * s (x) + x}  dx) - 5. 
        This is the same as R(s) = 5*log_{10} (e^x * d_H * int[0,x]  e^{-0.5 * s (x)} * e^x  dx) - 5. 
        d_H is the hubble distance and given above. The integral is carried out via the operation copied here from 
        the line of sight operator (~ sth. like self._smat.rmatvec(testarray).reshape(self.domain[0].shape) )
        
        Strategy: 
        1.) First take the input data from the correlated field (which is evaluated at a random position), scale it by -1/2 and exponentiate it
        2.) Multiply with the exponentiated redshift weights (e^x). Remember x=ln(1+z). 
        3.) Perform the integration via self._smat.rmatvec(testarray).reshape(self.domain[0].shape). Save it as a variable 'integral'
        4.) Return 5*log_{10} e^x d_H integral -5
        
        '''

        input_data = x.val.reshape(-1)


        ''' Step 1 : Scale input '''
        step_1_arr = np.exp(-1/2 * input_data)

        ''' Step 2 : Multiply with weights = integrand '''
        step_2_arr = step_1_arr * exponentiated_weights

        ''' Step 3 : Perform integration. Perform adjoint matrix vector multiplication and reshape array if mode!=Self.times '''
        if mode != self.TIMES:
            step_3_arr = self._smat.rmatvec(step_2_arr).reshape(self.domain[0].shape)
        else:
            step_3_arr = self._smat.matvec(step_2_arr)
        #for i,j in zip(step_2_arr,step_3_arr):
        #    print("input value: ",i, "turned to integral value", j)



        ''' Step 4 : Taking the log and other manipulations of integral values. return Result '''
        result = 5 * np.log10(exponentiated_weights * d_H * step_3_arr) - 5

        print("Response Operator result --->",result)

        plt.subplot(2, 2, 1)
        plt.plot(self.weights,step_2_arr,"b.")
        list = []
        for i in step_2_arr:
            if i<0:
                list.append(i)
        plt.plot(list,"r.")
        plt.title("integrand")

        plt.subplot(2, 2, 2)
        plt.plot(self.weights,step_3_arr,".",color="black")
        plt.title("integral")

        plt.subplot(2, 2, 3)
        plt.plot(self.weights,result, ".",color="lightblue")
        plt.title("result")

        if len(os.listdir("/Users/iason/PycharmProjects/Nifty/CHARM/figures"))!=0:
            numbas=[]
            for file in os.listdir("/Users/iason/PycharmProjects/Nifty/CHARM/figures"):
                numbas.append(file.split("_")[1][0])
            maxim = int(max(numbas))+1
            #plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/fig_"+str(maxim)+".png")
        else:
            pass
            #plt.savefig("/Users/iason/PycharmProjects/Nifty/CHARM/figures/fig_1.png")

        plt.clf()

        if mode != self.TIMES:
            return ift.Field(self._domain, result)
        else:
            return ift.Field(self._target, result)


def pow_spec(k):
    return np.nan_to_num((k)**(-4))
