import scipy.io
import numpy as np
from surmise.emulation import emulator

mat = scipy.io.loadmat(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\FayansFall2021.mat')
errmap = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\errmap.txt', delimiter=',', dtype=int)
inputs = np.loadtxt(r'C:\Users\moses\Desktop\git\surmise\research\emucomp\fayans_emucal\inputdata.csv', delimiter=',', dtype=object)

esave = mat['ESAVE']
theta = mat['XSAVE']
f = mat['FSAVE']

toterr = esave @ errmap
simpleerr = (toterr > 0.5)
wherefails = np.argwhere(simpleerr)
for r, c in wherefails:
    f[r, c] = np.nan

failtotals = simpleerr.sum(1)
topfailind = np.argpartition(failtotals, -50)[-50:]
completeinds = np.argwhere(simpleerr.sum(1) < 0.5).squeeze()
incompleteinds = np.argwhere(simpleerr.sum(1) > 0.5).squeeze()

# compile data
train_inds = np.hstack((np.random.choice(completeinds, 450, replace=False),
                        topfailind))
                        # np.random.choice(incompleteinds, 100, replace=False)))
ftrain = f[train_inds]
thetatrain = theta[train_inds]
test_inds = np.setdiff1d(np.arange(f.shape[0]), train_inds)
ftest = f[test_inds]
thetatest = theta[test_inds]

import time
start = time.time()
emu = emulator(inputs, thetatrain, np.copy(ftrain.T), method='PCGPwM',
               options={'xrmnan': 'all',
                        'thetarmnan': 'never',
                        'return_grad': True})
end = time.time()
print('time taken: ', end - start)

pred0 = emu.predict()
pred = emu.predict(inputs, thetatest)
print('Dimension compatibility: ', emu.predict().mean().shape == ftrain.T.shape)
predmean = pred.mean()

mse = (predmean - ftest.T)**2
frng = np.atleast_2d(np.nanmax(ftest, 0) - np.nanmin(ftest, 0)).T
rmse_x = np.sqrt(np.nanmean((predmean - ftest.T)**2 / frng, axis=1))
rmse_theta = np.sqrt(np.nanmean((predmean - ftest.T)**2 / frng, axis=0))


import scipy.stats as sps
class prior_fayans:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return sps.beta.logpdf(theta, 2, 2).sum(1).reshape((theta.shape[0], 1))

    def rnd(n):
        return sps.beta.rvs(2, 2, size=(n, 13))

from surmise.calibration import calibrator
np.set_printoptions(precision=3)
y = np.zeros(198)
yvar = np.ones(198)
cal = calibrator(emu=emu, y=y, yvar=yvar, x=inputs, thetaprior=prior_fayans,
                 method='directbayeswoodbury')

calpred = cal.predict()
posttheta = cal.theta.rnd(5000)
print(np.quantile(posttheta, q=(0.05, 0.95), axis=0).T)



#
import matplotlib.pyplot as plt
plt.style.use(['science','high-vis','grid'])
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.scatter(np.arange(198)+1, rmse_x, marker='x')
# ax.tick_params('both', labelsize=15)
# plt.yscale('log')
# plt.xlabel('Observables',fontsize=20)
# plt.ylabel('RMSE',fontsize=20)
# plt.tight_layout()
# plt.close()

fig, ax = plt.subplots(figsize=(6,6))
plt.imshow(np.isnan(ftrain), aspect='auto', cmap='gray', interpolation='none')
ax.tick_params('both', labelsize=15)
plt.ylabel('parameters', fontsize=20)
plt.xlabel('observables', fontsize=20)
plt.savefig('fayans_train.png', dpi=150)