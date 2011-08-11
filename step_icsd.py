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
        this_result = (1/4.0)*(
                source_term_one(z[j], r_sq, z=ub) -
                source_term_one(z[j], r_sq, z=lb) -
                source_term_two(z[j], ub=ub, lb=lb) +
                image_term(it=it, j=z[j], r_sq=r_sq, z=ub) - 
                image_term(it=it, j=z[j], r_sq=r_sq, z=lb)) 
        result[j,i] = this_result
    return result

step_icsd_forward_solution_matrix.__doc__ = \
        utils.BASIC_FORWARD_SOLUTION_DOCSTRING %\
        'the CSD is stepwise constant'

def integrand(z, r_sq=None, j=None):
    return (1/2.0)*(numpy.sqrt(r_sq + (j-z)**2) - numpy.abs(j - z))

def mirror_integrand(z, r_sq=None, j=None):
    return (1/2.0)*(numpy.sqrt(r_sq + (j+z)**2) - numpy.abs(j + z))

def source_term_one(j=None, r_sq=None, z=None):
    a = (z - j)*numpy.sqrt(r_sq + (j - z)**2)
    b = r_sq*numpy.log(numpy.sqrt(r_sq + (j - z)**2) - j + z)
    return a + b

def source_term_two(j, ub=None, lb=None):
    if numpy.average([ub, lb]) == j: # diagonal term
        # evaluate diagnal term explicitly, since it crosses zero.
        h = ub-lb
        c = h**2/2.0
    else:
        c_term = lambda z: -z*(z-2*j)*numpy.sign(j-z)
        c = c_term(ub) - c_term(lb)
    return c

def image_term(it=None, j=None, r_sq=None, z=None):
    a = (z + j)*numpy.sqrt(r_sq + (j + z)**2)
    b = r_sq*numpy.log(numpy.sqrt(r_sq + (j + z)**2) + j + z)
    c = -z*(z+2*j)
    return it*(a + b + c)

