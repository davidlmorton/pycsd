import itertools
import numpy

from pycsd import utils

def delta_icsd(voltage_traces, depths, diameter=500, sigma_above='inf'):
    ''' Docstring found below '''
    utils.check_icsd_args(voltage_traces, depths, diameter, sigma_above)

    # calculate the forward-solution matrix from the geometry provided.
    f = delta_icsd_forward_solution_matrix(depths, diameter, sigma_above)

    return utils.basic_icsd(f, voltage_traces)

delta_icsd.__doc__ = utils.BASIC_ICSD_DOCSTRING % 'delta'

def delta_icsd_forward_solution_matrix(depths, diameter, sigma_above):
    ''' Docstring found below '''
    num_depths, h, z, r_sq, it = utils.get_forward_solution_variables(
            depths, diameter, sigma_above)

    result = numpy.matrix(numpy.empty((num_depths, num_depths), 
            dtype=numpy.float64))
    for i, j in itertools.product(range(num_depths), range(num_depths)):
        result[j,i] = (h/2.0)*(
                     (numpy.sqrt((z[j] - z[i])**2 + r_sq) - abs(z[j] - z[i]))
                + it*(numpy.sqrt((z[j] + z[i])**2 + r_sq) - abs(z[j] + z[i])))
    return result
    

delta_icsd_forward_solution_matrix.__doc__ = \
        utils.BASIC_FORWARD_SOLUTION_DOCSTRING %\
        'the CSD consists of infinitely thin disks'
