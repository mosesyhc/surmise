import numpy as np

n = [50, 100, 250, 1000, 2500]
function = ['borehole', 'piston', 'otlcircuit', 'wingweight']
failrandom = ['True', 'False']
failfrac = [0.01, 0.05, 0.25]
method = ['GPEmGibbs', 'colGP', 'PCGPwM', 'GPy', 'PCGP_KNN', 'PCGP_benchmark', 'PCGP_BR']

combs = np.array(())
base = np.array(np.meshgrid(n, function, failrandom, failfrac, method)).T.reshape(-1, 5)
for i in range(10):
    rep = np.random.randint(1, 50000, size=120)
    reps = np.tile(rep, 7)
    comb = np.column_stack((base, reps))
    if i == 0:
        combs = comb.copy()
    else:
        combs = np.row_stack((combs, comb))
np.savetxt('params0.txt', combs[:2100], fmt='%s', delimiter='\t')
np.savetxt('params1.txt', combs[2100:4200], fmt='%s', delimiter='\t')
np.savetxt('params2.txt', combs[4200:6300], fmt='%s', delimiter='\t')
np.savetxt('params3.txt', combs[6300:], fmt='%s', delimiter='\t')
