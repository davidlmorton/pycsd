import numpy

from pycsd.delta_icsd import delta_icsd_forward_solution_matrix

def create_test_csd_data(kind, depths):
    """ 
    Create data like one of the three types used in figure 2 of:

    Pettersen et al. Journal of Neuroscience Methods (2006) 116-133

    Inputs: 
        kind: One of 'square', 'sine', or 'gauss'
        depths: 1D array of depths to sample functions at
    Returns:
        test_csd_data: CSD values (uA/mm**3) for each depth.  
    """
    generator_index = {'square': square_pulses,
                       'sine': sine_function,
                       'gauss': gaussian_function}

    return [generator_index[kind](depth) for depth in depths]

def create_potential_like_data(kind, depths, diameter=500, sigma_above='inf'):
    """ 
    Create potential-like data using the forward-solution, based on the
    CSD data created using "create_test_csd_data".

    Inputs: 
        kind: One of 'square', 'sine', or 'gauss' specifying the type of
              test CSD data to use.
        depths: The depths (in um) of the voltage_traces, with 0.0 being at the 
                surface of the neural tissue.  Positive values correspond
                to deeper recordings.  A list or 1D numpy array of floats.
                NOTE: depths are expected to be equally spaced from each other.
        diameter: The diameter (in um) of the model current disk.
        sigma_above: One of 'zero', 'same', or 'inf'.  
            'zero' -- Corresponding to the case where the media above the 
                    neural tissue has conductivity that is much less than 
                    the contuctivity within the tissue.
            'same' -- The conductivities are approximately the same.
            'inf' -- The media above the neural tissue has a much higher
                    conductivity than the neural tissue does.
    Returns:
        potential_like_data: The result of the forward solution, leaving out
                the conductivity of the neural tissue.  These values divided 
                by 0.3*1e-6 (S*um) will result in units of mV.
    """
    csd = numpy.matrix(create_test_csd_data(kind, depths))
    f = delta_icsd_forward_solution_matrix(depths, diameter, sigma_above)

    # multiply by f-matrix and return
    result = f*csd.T
    return numpy.array(result)

def square_pulses(z, ranges={(100, 400):1.0, (400, 1300):-1/3.0}):
    for r, value in ranges.items():
        if r[0] < z <= r[1]:
            return value
    return 0.0

def sine_function(z, wavelength=1000, z_range=(100, 1100)):
    if z_range[0] < z <= z_range[1]:
        return numpy.sin(numpy.pi*2.0*(z - z_range[0])/wavelength)
    return 0.0

def gaussian_function(z, mus=(300.0, 800.0), lambdas=(80.0, 250.0)):
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
