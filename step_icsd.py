import itertools
import numpy
from scipy import integrate

from pycsd import utils

def step_icsd(voltage_traces, depths, diameter=500, sigma_above='inf'):
    ''' Docstring found below '''
    utils.check_icsd_args(voltage_traces, depths, diameter, sigma_above)

    # calculate the forward-solution matrix from the geometry provided.
    f = step_icsd_forward_solution_matrix(depths, diameter, sigma_above)

    return utils.basic_icsd(f, voltage_traces)

step_icsd.__doc__ = utils.BASIC_ICSD_DOCSTRING % 'step'

def step_icsd_forward_solution_matrix(depths, diameter, sigma_above):
    ''' Docstring found below '''
    num_depths, h, z, r_sq, it = utils.get_forward_solution_variables(
            depths, diameter, sigma_above)

    result = numpy.matrix(numpy.empty((num_depths, num_depths), 
            dtype=numpy.float64))
    for i, j in itertools.product(range(num_depths), range(num_depths)):
        ub = z[i] + (h/2.0)
        lb = z[i] - (h/2.0)
        z_j = z[j]
        if i == j: 
            # not sure why, but we must use numerical integration to
            #  get result accurately for i==j.
            f = lambda x: integrand(x, r_sq=r_sq, z_j=z_j)
            f_mirror = lambda x: mirror_integrand(x, r_sq=r_sq, z_j=z_j)
            tissue_term = integrate.quad(f, lb, ub, epsrel=0.01)[0]
            mirror_term = integrate.quad(f_mirror, lb, ub, epsrel=0.01)[0]
            this_result = tissue_term + it*mirror_term
        else:
            this_result = (1/4.0)*(first_term(z_j=z_j, r_sq=r_sq, z=ub) - 
                               first_term(z_j=z_j, r_sq=r_sq, z=lb) + 
                               second_term(it=it, z_j=z_j, r_sq=r_sq, z=ub) - 
                               second_term(it=it, z_j=z_j, r_sq=r_sq, z=lb)) 
        result[j,i] = this_result
    return result

step_icsd_forward_solution_matrix.__doc__ = \
        utils.BASIC_FORWARD_SOLUTION_DOCSTRING %\
        'the CSD is stepwise constant'

def integrand(z, r_sq=None, z_j=None):
    return (1/2.0)*(numpy.sqrt(r_sq + (z_j-z)**2) - numpy.abs(z_j - z))

def mirror_integrand(z, r_sq=None, z_j=None):
    return (1/2.0)*(numpy.sqrt(r_sq + (z_j+z)**2) - numpy.abs(z_j + z))

def first_term(z_j=None, r_sq=None, z=None):
    a = (z - z_j)*numpy.sqrt(r_sq + (z_j - z)**2)
    log_arg = numpy.sqrt(r_sq + (z_j - z)**2) - z_j + z
    b = r_sq*numpy.log(numpy.sqrt(r_sq + (z_j - z)**2) - z_j + z)
    c = z*(z-2*z_j)*numpy.sign(z_j-z)
    return a + b + c

def second_term(it=None, z_j=None, r_sq=None, z=None):
    a = (z + z_j)*numpy.sqrt(r_sq + (z_j + z)**2)
    b = r_sq*numpy.log(numpy.sqrt(r_sq + (z_j + z)**2) + z_j + z)
    c = -z*(z+2*z_j)*numpy.sign(z_j+z)
    return it*(a + b + c)

