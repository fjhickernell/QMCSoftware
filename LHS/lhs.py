'''
Owen, A. B. (1994). Controlling Correlations in Latin Hypercube Samples. 
Journal of the American Statistical Association, 89(428), 1517. 
doi:10.2307/2291014
'''

from numpy import *
from scipy import stats
from scipy.linalg import solve_triangular

def lhs(n,d): 
    x = array([(random.permutation(n)+.5)/n for j in range(d)],dtype=float64).T
    return x

def lhs_rc(n,d):
    U = random.rand(n,d)
    trys = 0
    while True:
        Z = array([stats.norm.ppf((random.permutation(n)+1)/(n+1)) for j in range(d)],dtype=float64).T
        C = cov(Z.T)
        rank = linalg.matrix_rank(C)
        if rank==d: break
        trys += 1
        if trys>=10: raise Exception("C not full rank after many tries.")
    Q = linalg.cholesky(C)
    Qinv = solve_triangular(Q,eye(d),lower=True,check_finite=False)
    Zstar = Z@Qinv
    for j in range(d):
        u = U[:,j]
        z = Z[:,j]
        ranks = stats.rankdata(z).astype(int)-1
        args = argsort(u)
        U[:,j] = u[args[ranks]]

def lhs_rgs(n,d,niter):
    x = lhs(n,d)
    x -= .5 # remove mean
    for s in range(niter):
        # forward
        for j in range(1,d):
            for k in range(j):
                x[:,k] = _takeout(x[:,j],x[:,k])
            x[:,k] = (stats.rankdata(x[:,k])-.5)/n-.5#(argsort(x[:,k])+.5)/n#-.5 # use rankdata instead? 
        # backward
        for j in range(d-2,-1,-1):
            for k in range(d-1,j,-1):
                x[:,k] = _takeout(x[:,j],x[:,k])
            x[:,k] = (stats.rankdata(x[:,k])-.5)/n-.5#(argsort(x[:,k])+.5)/n#-.5 # use rankdata instead? 
    x += .5 # restore mean
    return x

def _takeout(x,y): # residuals when y is regressed onto x
    alpha_hat,beta_hat = _slr(x,y)
    eps = y-(alpha_hat+beta_hat*x)
    return eps

def _slr(x,y):
    xbar = mean(x)
    ybar = mean(y)
    beta_hat = sum((x-xbar)*(y-ybar))/sum((x-xbar)**2)
    alpha_hat = ybar-beta_hat*xbar
    return alpha_hat,beta_hat

def _get_corr_rho(x,d):
    corr = corrcoef(x.T)
    rho = sqrt(sum(tril(corr,-1)**2)/((d-1)*d/2))
    return corr,rho

def _test_single(n,d):
    set_printoptions(precision=2)
    class PTS(object): pass
    LHS = PTS()
    LHS.name = 'LHS'
    LHS.x = lhs(n,d)
    RGS = PTS()
    RGS.name = 'LHS-RGS'
    RGS.x = lhs_rgs(n,d,niter=1)
    RC = PTS()
    RC.name = 'LHS-RC'
    RC.x = lhs_rc(n,d)
    objs = [LHS,RC,RGS]
    shift = lambda s,t: str(s).replace('\n','\n'+('\t'*t))
    for obj in objs:
        obj.corr,obj.rho = _get_corr_rho(obj.x,d)
        print('%s\n\n\tx:\n\t\t%s\n\n\tcorr:\n\t\t%s\n\n\trho: %.1e\n'%\
            (obj.name,shift(obj.x,2),shift(obj.corr,2),obj.rho))
    if d!=2:
        print('d!=2 --> no single test plot produced.')
        return
    from matplotlib import pyplot
    fig,ax = pyplot.subplots(nrows=1,ncols=len(objs),figsize=(5*len(objs),5))
    for i,obj in enumerate(objs):
        ax[i].scatter(obj.x[:,0],obj.x[:,1])
        for j in range(n-1):
            ax[i].axhline(y=(j+1)/n)
            ax[i].axvline(x=(j+1)/n)
        ax[i].set_xlim([0,1])
        ax[i].set_ylim([0,1])
        ax[i].set_aspect(1)
        ax[i].set_title(obj.name)
    fig.tight_layout()
    fig.savefig('points.png')

def _test_multi(ses,niter,d,trials):
    dog = d
    ns = len(ses)
    rho_lhs = zeros((ns,trials),dtype=float)
    rho_rgs = zeros((ns,trials),dtype=float)
    print('sample size: ',end='',flush=True)
    for s in range(ns):
        n = ses[s]
        print('%d, '%n,end='',flush=True)
        if dog==False:
            d = n-1
        for t in range(trials):
            x_lhs = lhs(n,d)
            rho_lhs[s,t] = _get_corr_rho(x_lhs,d)[1]
            x_rgs = lhs_rgs(n,d,niter=niter)
            rho_rgs[s,t] = _get_corr_rho(x_rgs,d)[1]
    print()
    rho_lhs = rho_lhs.mean(1)
    rho_rgs = rho_rgs.mean(1)
    a_lhs,b_lhs = _slr(log(ses),log(rho_lhs))
    lhs_hat = exp(a_lhs+b_lhs*log(ses))
    a_rgs,b_rgs = _slr(log(ses),log(rho_rgs))
    rgs_hat = exp(a_rgs+b_rgs*log(ses))
    from matplotlib import pyplot
    fig,ax = pyplot.subplots(figsize=(8,5))
    ax.scatter(ses,rho_lhs,color='c',label='LHS')
    ax.plot(ses,lhs_hat,color='c',label=r'$\rho_{rms} \approx %.2fn^{%.2f}$'%(a_lhs,b_lhs))
    ax.scatter(ses,rho_rgs,color='m',label='LHS-RGS')
    ax.plot(ses,rgs_hat,color='m',label=r'$\rho_{rms} \approx %.2fn^{%.2f}$'%(a_rgs,b_rgs))
    ax.set_xscale('log',basex=2)
    ax.set_yscale('log',basey=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig('rho_%d.png'%dog)

if __name__ == '__main__':
    _test_single(
        n = 8, # samples size
        d = 2) # dimension. will only create a plot if d==2
    '''
    _test_multi(
        ses = 2**arange(3,6), # sample sizes
        niter = 7, # iterations for RGS
        d = False, # dimension. Set to False to use d=n-1 
        trials = 10) # number of trials to average over
    _test_multi(
        ses = 2**arange(3,10), # sample sizes
        niter = 7, # iterations for RGS
        d = 9, # dimension. Set to False to use d=n-1 
        trials = 10) # number of trials to average over
    '''
    