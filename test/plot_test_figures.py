import numpy
import pylab

from pycsd import delta_icsd as di
from pycsd import step_icsd as si
from pycsd.test import create_test_data as ctd

depths_100 = numpy.arange(100, 2600, 100)
depths_200 = numpy.arange(100, 2600, 200)
smooth_depths = numpy.arange(25, 2525, 25)

# create the sine wave test data
test_csd = ctd.create_test_csd_data('sine', smooth_depths)
test_pot = ctd.create_potential_like_data('sine', smooth_depths, diameter=100000,
        sigma_above='inf')

test_pot_100 = numpy.array([test_pot[i] for i in range(len(test_pot)) 
        if smooth_depths[i] in depths_100])
test_pot_200 = numpy.array([test_pot[i] for i in range(len(test_pot)) 
        if smooth_depths[i] in depths_200])

step_csd_100 = si.step_icsd(test_pot_100, depths_100, diameter=100000, 
    sigma_above='inf')
step_csd_200 = si.step_icsd(test_pot_200, depths_200, diameter=100000, 
    sigma_above='inf')

delta_csd_100 = di.delta_icsd(test_pot_100, depths_100, diameter=100000,
    sigma_above='inf')
delta_csd_200 = di.delta_icsd(test_pot_200, depths_200, diameter=100000,
    sigma_above='inf')

pylab.plot(test_csd, -smooth_depths)
pylab.plot(step_csd_100, -depths_100)
pylab.plot(step_csd_200, -depths_200)
pylab.plot(delta_csd_100, -depths_100, linestyle=':', marker='o')
pylab.plot(delta_csd_200, -depths_200, linestyle=':', marker='o')

pylab.show()


