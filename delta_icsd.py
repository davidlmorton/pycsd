import itertools
import numpy

from pycsd import utils

def delta_icsd(voltage_traces, depths, diameter=500, sigma_above='inf'):
    '''
        Estimate the CSD using the delta-iCSD method as described in:

    K.H. Pettersen et al. Journal of Neuroscience Methods 154 (2006) 116-133

    Inputs:
        voltage_traces: The recorded potential (in mV).  A 2D numpy 
                array of floats.  The array should be of shape (m x n) where 
                m is the number of recording depths, and n is the number of 
                voltage samples.  If this is a 1D numpy array or a list, it
                is treated as a single voltage reading and must therefore 
                be the same length as <depths>.
        depths: The depths (in um) of the voltage_traces, with 0.0 being at the 
                surface of the neural tissue.  Positive values correspond
                to deeper recordings.  A list or 1D numpy array of floats.
        diameter: The diameter (in um) of the model current disk.
        sigma_above: One of 'zero', 'same', or 'inf'.  
            'zero' -- Corresponding to the case where the media above the 
                    neural tissue has conductivity that is much less than 
                    the contuctivity within the tissue.
            'same' -- The conductivities are approximately the same.
            'inf' -- The media above the neural tissue has a much higher
                    conductivity than the neural tissue does.
    Returns:
        csd_per_cortical_conductivity :  A numpy array of the same shape as 
                <voltage_traces>.  The corresponding units are: mV/(um**2).  
                That is, this value times the cortical conductivity is the 
                full csd estimate.
    '''
    utils.check_icsd_args(voltage_traces, depths, diameter, sigma_above)

    # calculate the forward-solution matrix from the geometry provided.
    f = delta_icsd_forward_solution_matrix(depths, diameter, sigma_above)

    # invert the forward-solution matrix
    f_inv = numpy.linalg.inv(f)

    # multiply inverse and voltage_traces
    c = numpy.matrix(numpy.empty(voltage_traces.shape, dtype=numpy.float64))
    if len(voltage_traces.shape) == 2:
        for i in range(voltage_traces.shape[1]):
            c[:,i] = f_inv * numpy.matrix(voltage_traces[:,i]).T
    else:
        c = f_inv * numpy.matrix(voltage_traces).T
    return numpy.array(c)

def delta_icsd_forward_solution_matrix(depths, diameter, sigma_above):
    '''
        Return the forward-solution matrix for CSD estimation assuming that the
    CSD is restricted to infinitely thin disks of the given diameter.
    Inputs:
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
        f: The forward-solution matrix.  A 2D numpy matrix of shape (m x m) 
                where m is the length of <depths>
    '''
    utils.check_forward_solution_matrix_args(depths, diameter, sigma_above)

    # ensure depths are equally spaced
    num_depths = len(depths)
    h = depths[1]-depths[0] # electrode separation
    z = [depths[0] + j*h for j in range(num_depths)]
    r_sq = (diameter/2.0)**2 

    # term introduced by the method of images based on the sigma_above value.
    image_term_index = {'zero':1.0, 'same':0.0, 'inf':-1.0}
    it = image_term_index[sigma_above]

    result = numpy.matrix(numpy.empty((num_depths, num_depths), 
            dtype=numpy.float64))
    for i, j in itertools.product(range(num_depths), range(num_depths)):
        result[j,i] = (h/2.0)*(
                     (numpy.sqrt((z[j] - z[i])**2 + r_sq) - abs(z[j] - z[i]))
                + it*(numpy.sqrt((z[j] + z[i])**2 + r_sq) - abs(z[j] + z[i])))
    return result

