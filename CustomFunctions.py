import numpy as np
import scipy as sc
import nifty8 as ift

def response_operator(x_j,signal_field):
    #d_h is the hubble distance d_h = c/H_0
    #rho_0 is the critical density rho_0 = 3H_0^2/(8piG)
    #signal_field is a correlated field model signal_field = signal_field (x)  = CorrFModel (x)
    d_h = 1
    rho_0 = 1

    # the signal field is to be inferred.
    # the integral values of the response operator. The data is stored in a Field.

    # TODO do you have to use integrate.simpson here ?
    integral_val = sc.integrate.quad(lambda x: np.exp(-0.5*signal_field+x),0,x_j)

    # see
    #R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)

    return 5*np.log10(np.exp(x_j)*d_h*integral_val)-5

class PowerSpectrum:
    # alpha is the strength punishement parameter
    def __init__(self, alpha, k):
        self.k = k
        self.alpha = alpha

    def func(self):
        return self.alpha / (self.k ** 4)

def power_spectrum(k):
    return 0.4 / (k ** 4)

def response_operator2(signal_domain, data_domain, redshifts, distance_moduli, background_cosmology, initial_point):
    return None


class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)


class SampleIntegrationOfCorrelatedFieldOperator(ift.Operator):

    def __init__(self, datadomain, x_natural, signal_space_domain,x_linspace,signal_space_as_cf):

        self.datadomain = datadomain

        self._target = ift.DomainTuple.make(signal_space_as_cf.target)
        self._domain = ift.DomainTuple.make(datadomain)

        self.x_natural = x_natural
        self.signal_domain = signal_space_domain
        self.x = x_linspace # for general operations a linspace subset of x_natural

    def apply(self):
        d_h = 1
        rho_0 = 1

        # integral integrand as a field; initialize empty list to store integral values
        integrand_field = ift.Field(ift.DomainTuple.make(self.signal_domain,), val = np.exp(-0.5* self.signal_space_as_cf + self.x_natural))
        integral_values = []

        for j in range(0,len(self.x_natural)):
            # last pixel position in signal (space) domain, what is smaller than the upper limit of the integral i.e. x_natural[j]
            last_pixel_pos = np.where(self.x < self.x_natural[j])
            print("last pixel position should be int" , last_pixel_pos)
            y1 = integrand_field[0:last_pixel_pos]
            x1 = self.x[0:last_pixel_pos]
            # check if the distance between the natural x_j and the value from the linspace x array where pos = last index +1 is small
            if abs(self.x[last_pixel_pos + 1] - self.x_natural) < 1e-16:
                last_pixel_pos += 1
                integral_values.append(np.exp(self.x_natural[j])*sc.integrate.simpson(y1,x1))
            else:
                # extrapolate to integral limit
                y_point = (self.x_natural[j]-self.x[last_pixel_pos])*(integrand_field[last_pixel_pos+1]-integrand_field[last_pixel_pos])/(self.x[last_pixel_pos+1]-self.x[last_pixel_pos]) + integrand_field[last_pixel_pos]
                x_point = self.x_natural[j]
                y2 = np.append(y1,y_point)
                x2 = np.append(x1,x_point)
                integral_values.append(np.exp(self.x_natural[j])*sc.integrate.simpson(y2,x2))

        integral_values = np.array(integral_values)
        # Define a field in data space to store data vom integration
        integral_field = ift.Field(ift.DomainTuple.make(self.datadomain),val=integral_values)
        return 5 * np.log10(d_h * integral_field) - 5


class TestOperator(ift.Operator):



    def __init__(self,domain, target, x_natural_space, x_natural_linspace):
        self._target = ift.DomainTuple.make(target)  # target of response operator: Data Space
        self._domain = ift.DomainTuple.make(domain)  # domain of response operator: Signal Space

        self.x_natural = x_natural_space
        self.x = x_natural_linspace

    def apply(self, corr_field):
        d_h = 1
        # integrand of response as a field; initialize empty list to store integral values
        # val=np.exp(-0.5 * corr_field + self.x_natural) = np.exp(-0.5 * corr_field)*np.exp(self.x_natural)

        scaled_correlated_field = []
        print("x natural",self.x_natural)
        for x_j in self.x_natural:
            #scaled_correlated_field.append(np.exp(-0.5 * corr_field)*np.exp(x_j))
            scaled_correlated_field.append(2)

        integrand_field = ift.Field(self._domain, val=np.array(scaled_correlated_field))
        integral_values = []

        print("not implemented? ",integrand_field[0:5])

        # calculate integral value for each redshift data point
        for j in range(0,len(self.x_natural)):
            # last pixel position in signal space domain that is smaller than the upper limit of the integral i.e. x_natural[j]
            last_pixel_pos = np.where(self.x < self.x_natural[j])[-1][-1]
            print("last pixel position should be int" , last_pixel_pos)
            # define y and x samples to simpson-integrate over
            y1 = integrand_field[0:last_pixel_pos]
            x1 = self.x[0:last_pixel_pos]
            # normally calculate the integral if the distance between the natural x [j] and the value from the linspace x array where pos = last index +1 is small
            if abs(self.x[last_pixel_pos + 1] - self.x_natural[j]) < 1e-16:
                last_pixel_pos += 1
                integral_values.append(np.exp(self.x_natural[j]) * sc.integrate.simpson(y1, x1))
            # else extrapolate to integral limit
            else:
                y_point = (self.x_natural[j] - self.x[last_pixel_pos]) * (
                            integrand_field[last_pixel_pos + 1] - integrand_field[last_pixel_pos]) / (
                                      self.x[last_pixel_pos + 1] - self.x[last_pixel_pos]) + integrand_field[
                              last_pixel_pos]
                x_point = self.x_natural[j]
                y2 = np.append(y1, y_point)
                x2 = np.append(x1, x_point)
                integral_values.append(np.exp(self.x_natural[j]) * sc.integrate.simpson(y2, x2))

            integral_values = np.array(integral_values)
            # Define a field in data space to store data vom integration
            integral_field = ift.Field(self._target, val=integral_values)
            return 5 * np.log10(d_h * integral_field) - 5


class LineOfSightResponse:

    def __init__(self,signal_space,StartArray,EndArray):
        self.signal_space = signal_space
        self.StartArray = StartArray
        self.EndArray = EndArray



    def apply(self,x):
        LOSOperator = ift.LOSResponse(self.signal_space, self.StartArray,self.EndArray)
        return LOSOperator @ x
