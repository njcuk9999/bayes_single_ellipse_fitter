"""
Description of program
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import emcee
import corner
from neil_gen_functions import refreshing_msg


# ==============================================================================
# Define variables
# ==============================================================================
path = './'
dataname = 'test.fits'
nruns = 1000
nwalkers = 100
nburn = 300

# iterator keep here
iterator = 1


# ==============================================================================
# Define functions
# ==============================================================================
class mcmc_results:
    """
    Prepare the MCMC results in a more usable format
    """

    def __init__(self, sampler, p_keys):
        self.param_keys = p_keys
        self.sampler = sampler
        self.flatchain = sampler.flatchain
        # define empty param lists
        self.lower_bestfit_params = []
        self.best_fit_params = []
        self.best_fit_lower_errors = []
        self.best_fit_upper_errors = []
        self.upper_bestfit_params = []
        # get best fit params
        self.bestfitparams()

    def __call__(self):
        """
        Call method for mcmc results: prints best fit parameters
        """
        print self.getstring(ext='\n\t')

    def getstring(self, keys=None, latex=False, ext=''):
        if keys is None:
            keys = self.param_keys
        string = "\n\n Best fit parameters: "
        for ix in range(len(keys)):
            args = [keys[ix], self.best_fit_params[ix],
                    self.best_fit_upper_errors[ix],
                    self.best_fit_lower_errors[ix]]
            string += ext
            if latex:
                lstring = '  {0} $=({1:.3e})^{{+({2:.3e})}}_{{-({3:.3e})}}$'
                string += lstring.format(*args)
            else:
                string += ' {0} = ({1:.3e}) +({2:.3e}) -({3:.3e})'.format(*args)
        return string

    def bestfitparams(self):
        """
        Work out the best fit parameters based on the 1 sigma percentiles of
        each flat chain
        """
        self.lower_bestfit_params = []
        self.best_fit_params = []
        self.best_fit_lower_errors = []
        self.best_fit_upper_errors = []
        self.upper_bestfit_params = []
        for p in range(len(self.param_keys)):
            f = self.flatchain[:, p]
            # get 1 sigma percentile values
            lower = np.percentile(f, 16)
            best = np.percentile(f, 50)
            upper = np.percentile(f, 84)
            # append these to arrays
            self.lower_bestfit_params.append(lower)
            self.best_fit_params.append(best)
            self.upper_bestfit_params.append(upper)
            self.best_fit_lower_errors.append(best - lower)
            self.best_fit_upper_errors.append(upper - best)


def make_model(ps, xs, ys, keys):
    """
    Make an ellipse model (value of 1 where ellipse is and 0 where
    ellipse is not)
    :param ps: list, list of parameters in order of dictionary i.e.
               params.values()

               keys are as follows:
                    a, b = width and height of the ellipse (in pixels)
                    x0, y0 = pixel center
                    t = rotation angle of the ellipse (in radians)

    :param xs: array, x coordinates for every pixel of the image
    :param ys: array, y coordinates for every pixel on the image
    :param keys: list of strings, dictionary keys, i.e. params.keys()
    :return:
    """
    x0, y0 = ps[keys.index('x0')], ps[keys.index('y0')]
    a, b = ps[keys.index('a')], ps[keys.index('b')]
    t = ps[keys.index('t')]
    xx = xs - x0
    yy = ys - y0
    part1 = ((1/a)**2) * ((xx*np.cos(t) + yy*np.sin(t))**2)
    part2 = ((1/b)**2) * ((xx*np.sin(t) - yy*np.cos(t))**2)
    ellipsemask = part1 + part2 <= 1
    return np.array(ellipsemask, dtype=float)


def lnprior(ps, pv, pk):
    """
    Work out priors (flat priors based on extremes of grid_params)
    :param ps: list, instance of parameters
               i.e. for permutation0 = [param1, param2, ..., paramN]
               this should be in the order of the initial dictionary
               i.e. dictionary.values()
    :param pv: list of all possible values for param1, param2, ..., paramN in
               order of ps (i.e. dictionary.values())
    :param pk: list of parameter keys for param1, param2, ..., paramN in order
               of ps (i.e. dictionary.keys())
    :return:
    """
    for pp in range(len(ps)):
        gp = pv[pp]
        if not (min(gp) <= ps[pp] <= max(gp)):
            return -np.inf

    if ps[pk.index('a')] < ps[pk.index('b')]:
        return -np.inf
    return 0.0


def lnprob(ps, d_x, d_y, d_f, d_ef, pdict):
    """
    work out the log probability of a model with params ps matching the data
    :param ps: list, instance of parameters
               i.e. for permutation0 = [param1, param2, ..., paramN]
    :param d_x: array, the wavelength values for the data
    :param d_y: array, the flux values for data
    :param d_ey: array, the flux uncertainty values for data
    :param d_m: array, boolean mask (create by data) so models match datashape
    :param pvals: list of all possible parameter values for all ps values
                  in order of ps
    :param pkeys: list of parameter keys for all ps values in order of ps
    :param total: the total number of runs
    :return:
    """

    # print progress
    global iterator
    refreshing_msg('\t Processing {0}\t\t\t'.format(iterator))
    iterator += 1
    # calculate log prioer probability
    prior = lnprior(ps, pdict.values(), pdict.keys())
    # if prior is negatively infinite return it
    if prior == -np.inf:
        return -np.inf
    # if it isn't then calculate the posterior
    # get model fluxes from the interpolation
    modely = make_model(ps, d_x, d_y, pdict.keys())
    # calculate where the model is nan and mask
    nanmask = (modely == modely)
    if len(modely[nanmask]) == 0:
        return -np.inf
    # calculate the difference in data and model
    diff = d_f - modely
    # work out the probability
    # p = -np.dot(diff, np.dot(icov_y,diff))/2.0
    p = -0.5 * (np.nansum((diff / d_ef) ** 2))
    # p = prob_from_slices(d_x, d_y, d_ey, modely, xslices=slices)
    # make sure p is not nan
    if np.isnan(p):
        return -np.inf
    # return the probability (boils down to a gaussian)
    return p


def mcmcrun(gridparams, runs, walkers, burn, dim):
    """
    Bayesian run
    :param gridparams: dictionary, each element containing the bounds of one
                       of the parameters to vary

               keys are as follows:
                    a, b = width and height of the ellipse (in pixels)
                    x0, y0 = pixel center
                    t = rotation angle of the ellipse (in radians)

    :param runs: number of runs to do of the emcee code
    :param walkers: number of walkers to use for the emcee code
    :param burn: number of runs to burn before actual fitting begins
    :param dim: number of dimensions (in this case 5: a, b, x0, y0, t)
    :return:
    """

    # select a random starting point for each walker in the parameter space
    jx, p0 = 0, []
    while jx < walkers:
        p0i = []
        for ix in gridparams.values():
            p0i.append(np.random.choice(ix))
            # p0i.append(np.random.uniform(np.min(ix), np.max(ix)))
        p0.append(p0i)
        jx += 1

    # setup the emcee Ensemble Sampler
    args = [datax, datay, dataf, dataef, gridparams]
    sampler = emcee.EnsembleSampler(walkers, dim, lnprob, args=args, threads=8)
    # burn if burn is positive
    if burn > 0:
        pos, prob, state = sampler.run_mcmc(p0, burn)
    else:
        pos = p0
    sampler.reset()
    # run full MCMC fit
    sampler.run_mcmc(pos, runs)
    # return sampler
    return sampler


# ==============================================================================
# Start of code
# ==============================================================================
if __name__ == "__main__":
    # load data
    print '\n Loading data...'
    data = fits.getdata(path + dataname)
    # set up x pixels and y pixels
    xlen, ylen = len(data), len(data[0])
    x = np.arange(0.0, xlen, 1.0)
    y = np.arange(0.0, ylen, 1.0)
    datax = np.tile(x, len(y)).reshape(len(x), len(y))
    datay = np.repeat(y, len(x)).reshape(len(x), len(y))
    dataf = data
    # dataef = np.sqrt(dataf)
    dataef = np.ones((len(dataf), len(dataf[0])))
    # --------------------------------------------------------------------------
    # set up priors
    pspace = dict()
    # map out parameter space of central x pixel
    # this assumes the center of the ellipse is within the middle 20% of the
    # image.
    # WARNING THIS MAY NOT BE TRUE and depends on the input image
    pspace['x0'] = np.arange(0.4*xlen, 0.6*xlen, 1.0)
    # pspace['x0'] = np.arange(0.0, xlen, 1.0)
    # map out parameter space of central y pixel
    pspace['y0'] = np.arange(0.4*ylen, 0.6*ylen, 1.0)
    # pspace['y0'] = np.arange(0.0, ylen, 1.0)
    # map out parameter space of width (x direction) in pixels
    pspace['a'] = np.arange(1.0, xlen, 1.0)
    # map out parameter space of height (y direction) in pixels
    pspace['b'] = np.arange(1.0, ylen, 1.0)
    # map out parameter space of rotation (in radians)
    pspace['t'] = np.arange(0, np.pi)

    # --------------------------------------------------------------------------
    print 'Running MCMC...'
    samples = mcmcrun(pspace, nruns, nwalkers, nburn, len(pspace))
    results = mcmc_results(samples, pspace.keys())
    mask = make_model(results.best_fit_params, datax, datay, pspace.keys())
    mask = np.array(mask, dtype=bool)
    corner.corner(samples.flatchain, bins=100, labels=pspace.keys(),
                  show_titles=True)
    fig1, frame = plt.subplots(ncols=2, nrows=1)
    frame[0].imshow(data, origin='lower')
    frame[1].imshow(data, origin='lower')
    frame[1].scatter(datax[mask], datay[mask], color='k', alpha=0.5)
    plt.show()
    plt.close()

# ==============================================================================
# End of code
# ==============================================================================
