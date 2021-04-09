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
        if trys>=10: raise Exception("C not full rank. d=%d but most recent rank(C)=%d."%(d,rank))
    Q = linalg.cholesky(C)
    Qinv = solve_triangular(Q,eye(d),lower=True,check_finite=False)
    Zstar = Z@Qinv
    for j in range(d):
        u = U[:,j]
        z = Z[:,j]
        ranks = stats.rankdata(z).astype(int)-1
        args = argsort(u)
        U[:,j] = u[args[ranks]]
    P = array([random.permutation(n) for j in range(d)],dtype=float64).T
    x = (P+1-U)/n
    return x

def lhs_rgs(n,d,niter=7):
    x = lhs(n,d)
    x -= .5 # remove mean
    for s in range(niter):
        # forward
        for j in range(1,d):
            for k in range(j):
                x[:,k] = _takeout(x[:,j],x[:,k])
            x[:,k] = (stats.rankdata(x[:,k])-.5)/n-.5
        # backward
        for j in range(d-2,-1,-1):
            for k in range(d-1,j,-1):
                x[:,k] = _takeout(x[:,j],x[:,k])
            x[:,k] = (stats.rankdata(x[:,k])-.5)/n-.5
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
    rho = sqrt(sum((corr-eye(d))**2)/((d-1)*d))
    return corr,rho

class _PTS(object): pass

pbar = lambda: print('\n%s\n'%(50*'~'))

def _test_single(gens,names,colors,n,d):
    set_printoptions(precision=2)
    objs = []
    for gen,name in zip(gens,names):
        obj = _PTS()
        obj.name = name
        obj.x = gen(n,d)
        objs.append(obj)
    shift = lambda s,t: str(s).replace('\n','\n'+('\t'*t))
    for obj in objs:
        obj.corr,obj.rho = _get_corr_rho(obj.x,d)
        print('%s\n\n\tx:\n\t\t%s\n\n\tcorr:\n\t\t%s\n\n\trho: %.1e\n'%\
            ('LHS-%s'%obj.name,shift(obj.x,2),shift(obj.corr,2),obj.rho))
    if d!=2:
        print('d!=2 --> no single test plot produced.')
        return
    from matplotlib import pyplot
    fig,ax = pyplot.subplots(nrows=1,ncols=len(objs),figsize=(5*len(objs),5))
    for i,(obj,color) in enumerate(zip(objs,colors)):
        ax[i].scatter(obj.x[:,0],obj.x[:,1],color=color)
        for j in range(n-1):
            ax[i].axhline(y=(j+1)/n,color='k')
            ax[i].axvline(x=(j+1)/n,color='k')
        ax[i].set_xlim([0,1])
        ax[i].set_ylim([0,1])
        ax[i].set_aspect(1)
        ax[i].set_title('LHS-%s'%obj.name)
        #ax[i].set_xlabel("$X^1$")
        #ax[i].set_ylabel('$X^2$')
    fig.tight_layout()
    fig.savefig('points.png')
    pbar()

def _test_multi(gens,names,colors,ses,d,trials):
    dog = d
    ns = len(ses)
    objs = []
    for gen,name in zip(gens,names):
        obj = _PTS()
        obj.name = name
        obj.rhos = zeros((ns,trials),dtype=float)
        print('%15s: n = '%('LHS-%s'%obj.name),end='',flush=True)
        for s in range(ns):
            n = ses[s]
            print('%d, '%n,end='',flush=True)
            if dog==False:
                d = n-1
            for t in range(trials):
                x = gen(n,d)
                obj.rhos[s,t] = _get_corr_rho(x,d)[1]
        obj.rhos_mus = obj.rhos.mean(1)
        obj.alpha_hat,obj.beta_hat = _slr(log(ses),log(obj.rhos_mus))
        obj.lf = exp(obj.alpha_hat+obj.beta_hat*log(ses))
        objs.append(obj)
        print()
    from matplotlib import pyplot
    fig,ax = pyplot.subplots(figsize=(8,5))
    for obj,color in zip(objs,colors):
        ax.scatter(ses,obj.rhos_mus,color=color)
        for t in range(trials): ax.scatter(ses,obj.rhos[:,t],color=color,alpha=.5,marker='x')
        ax.plot(ses,obj.lf,color=color,label=r'$\rho_{%s} \approx %.2fn^{%.2f}$'%(obj.name,exp(obj.alpha_hat),obj.beta_hat))
    ax.set_xscale('log',basex=10)
    ax.set_yscale('log',basey=10)
    ax.set_xticks(ses)
    ax.set_xticklabels(['%d'%s for s in ses])
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("RMS Correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig('rho_%d.png'%dog)
    pbar()

if __name__ == '__main__':
    gens,names,colors = [lhs_rc,lhs_rgs],['RC','RGS'],['c','m']
    _test_single(gens,names,colors,
        n = 8, # samples size
        d = 2) # dimension. will only create a plot if d==2
    _test_multi(gens,names,colors,
        ses = array([10,20,30,50,100,150]),#,250,500]), # sample sizes
        d = False, # dimension. Set to False to use d=n-1 
        trials = 4) # number of trials to average over
    _test_multi(gens,names,colors,
        ses = array([10,20,30,100,150,250,500]), # sample sizes
        d = 9, # dimension. Set to False to use d=n-1 
        trials = 4) # number of trials to average over
    