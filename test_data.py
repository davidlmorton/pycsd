import numpy

from pycsd.delta_icsd import delta_icsd_forward_solution_matrix

def test_csd_data(kind, depths):
    """ This function creates a one of three test CSD

    Inputs: 
    Kind      : Type of csd, 3 options square, sine, gauss
    depths    : array of depths
    Returns:
    test_csd  
    """
    generator_index = {'square': square_pulses,
                       'sine': sine_function,
                       'gauss': gaussian_function}

    return [generator_index[kind](depth) for depth in depths]

def test_potential_data(kind, depths, diameter):
    csd = numpy.matrix(test_csd_data(kind, depths))
    f = numpy.matrix(delta_icsd_forward_solution_matrix(len(depths), depths[0], 
            depths[1]-depths[0], diameter)) 
    # multiply by f-matrix and return
    result = f*csd.T
    return numpy.array(result)


def square_pulses(z, ranges={(100, 400):1.0, (400, 1300):-1/3.0}):
    '''
    A function which returns the value of a multiple square pulses function.
    '''
    for r, value in ranges.items():
        if r[0] < z <= r[1]:
            return value
    return 0.0

def sine_function(z, wavelength=1000, z_range=(100, 1100)):
    '''
    Returns the sine function with given wavelength within the z_range, zero outside the range.
    '''
    if z_range[0] < z <= z_range[1]:
        return numpy.sin(numpy.pi*2.0*(z - z_range[0])/wavelength)
    return 0.0

def gaussian_function(z, mus=(300.0, 800.0), lambdas=(80.0, 250.0)):
    '''
    Returns the difference of gaussian functions
    '''
    e = numpy.exp

    normalizing_factor = 1.0/numpy.sqrt(2.0*numpy.pi)

    result = 0.0
    for i in range(len(mus)):
        if i == 0:
            sign_factor = 1.0
        else:
            sign_factor = -1.0

        result += sign_factor * e(-(z - mus[i])**2/(2*lambdas[i]**2))/lambdas[i]

    return normalizing_factor * result
