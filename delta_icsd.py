import itertools
import numpy

def delta_icsd(voltage_traces, first_electrode_depth=0.0, 
        electrode_separation=25.0, diameter=500):
    '''
Estimate the CSD using the delta-iCSD method as described in:

K.H. Pettersen et al. Journal of Neuroscience Methods 154 (2006) 116-133

Inputs:
    voltage_traces        : The recorded potential (in mV).  A 2D numpy array 
                            of floats.  The array should be of shape (m x n) 
                            where m is the number of recording depths, and n is
                            the number of voltage samples.
    first_electrode_depth : The depth (in um) of the most superficial electrode.
    electrode_separation  : The separation (in um) between the electrodes.
    diameter              : The diameter (in um) of the model current disk.
Returns:
    csd_per_cortical_conductivity :  A numpy array of the same shape as 
                                     <voltage_traces> of floats.  The 
                                     corresponding units are: mV/(um**2).
                                     That is, this value times the cortical
                                     conductivity is the full csd estimate.
    '''
    # calculate the forward-solution matrix from the geometry provided.
    f = delta_icsd_forward_solution_matrix(len(voltage_traces), 
            first_electrode_depth, electrode_separation, diameter)
    # invert the forward-solution matrix
    f_inv = numpy.matrix(numpy.linalg.inv(f))
    # multiply inverse and voltage_traces
    c = numpy.matrix(numpy.empty(voltage_traces.shape, dtype=numpy.float64))
    for i in range(voltage_traces.shape[1]):
        c[:,i] = f_inv * numpy.matrix(voltage_traces[:,i]).T
    return numpy.array(c)

def delta_icsd_forward_solution_matrix(num_depths, first_electrode_depth, 
        electrode_separation, diameter):
    '''
    Calculates the forward-solution matrix for CSD estimation assuming that the
CSD is restricted to infinitely thin disks of radius R.
    '''
    r = diameter/2.0
    h = electrode_separation
    z1 = first_electrode_depth
    z = [z1 + j*h for j in range(num_depths)]
    result = numpy.empty((num_depths, num_depths), dtype=numpy.float64)
    for i, j in itertools.product(range(num_depths), range(num_depths)):
        result[j][i] = (h/2.0)*(
                       (numpy.sqrt((z[j] - z[i])**2 + r**2) - abs(z[j] - z[i]))
                     - (numpy.sqrt((z[j] + z[i])**2 + r**2) - abs(z[j] + z[i])))
    return result

