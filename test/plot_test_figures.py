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
test_pot = ctd.create_potential_like_data('sine', smooth_depths, diameter=5000,
        sigma_above='inf')

test_pot_100 = numpy.array([test_pot[i] for i in range(len(test_pot)) 
        if smooth_depths[i] in depths_100])
test_pot_200 = numpy.array([test_pot[i] for i in range(len(test_pot)) 
        if smooth_depths[i] in depths_200])

step_csd_100 = si.step_icsd(test_pot_100, depths_100, diameter=5000, 
    sigma_above='inf')
step_csd_200 = si.step_icsd(test_pot_200, depths_200, diameter=5000, 
    sigma_above='inf')

delta_csd_100 = di.delta_icsd(test_pot_100, depths_100, diameter=5000,
    sigma_above='inf')
delta_csd_200 = di.delta_icsd(test_pot_200, depths_200, diameter=5000,
    sigma_above='inf')

pylab.plot(smooth_depths, test_csd)
pylab.plot(depths_100, step_csd_100, drawstyle='steps-mid')
pylab.plot(depths_200, step_csd_200, drawstyle='steps-mid')
pylab.plot(depths_100, delta_csd_100, linestyle=':', marker='o')
pylab.plot(depths_200, delta_csd_200, linestyle=':', marker='o')

pylab.show()


