import numpy as np
import scipy.stats as sps
from pyDOE import lhs
from testfunc_wrapper import TestFunc
from emu_single_test_pcs import single_test

# Number of input locations
nx = 15

models = ['piston'] # 'borehole', 'otlcircuit', 'wingweight',
emulator_methods = ['PCGPwM']# , 'GPy'] #, 'PCGP_KNN', 'PCGP_BR'] #

func_caller = TestFunc(models[0]).info
function_name = func_caller['function']
xdim = func_caller['xdim']
thetadim = func_caller['thetadim']

np.random.seed(0)
x = sps.uniform.rvs(0, 1, (nx, xdim))
np.random.seed()

testtheta = np.random.uniform(0, 1, (1000, thetadim))
theta = lhs(thetadim, 500)

model = func_caller['nofailmodel']
f = model(x, theta)

f1 = f.copy()
f1[:,250:] = np.nan
f2 = f.copy()
f2 = f2[:,:250]

offset = f2.mean(1)
scale = f2.std(1)
fs = ((f2.T - offset) / scale).T
U, S, _ = np.linalg.svd(fs, full_matrices=False)
Sp = S**2 - 0.001
Up = U[:, Sp > 0]
extravar = np.nanmean((fs.T - fs.T @ Up @ Up.T) ** 2, 0) * (scale ** 2)

standardpcinfo1 = {'offset': offset,
                  'scale': scale,
                  'fs': np.vstack((fs.T, np.full((250, 15), np.nan))),
                  'U': U,
                  'S': S,
                  'extravar': extravar
                  }

standardpcinfo2 = {'offset': offset,
                  'scale': scale,
                  'fs': fs.T,
                  'U': U,
                  'S': S,
                  'extravar': extravar
                  }

emu1 = single_test('PCGPwM', x, theta, f1, model, testtheta,
            'piston', 500, 'none', 'none', 10, 0, '')#, stdfinfo=standardpcinfo1),# skip_std=True, caller=func_caller)

emu2 = single_test('PCGPwM', x, theta[:250], f2, model, testtheta,
            'piston', 250, 'none', 'none', 10, 0, '')#, stdfinfo=standardpcinfo2)
