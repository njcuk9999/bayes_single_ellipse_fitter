# bayes single ellipse fitter

Fits an ellipse to an image (i.e. a galaxy) assumes there is only one ellipse in the image

This uses a very simple baye code (assuming a gaussian distribution and that all 
uncertainties are independent - hence collapsing the inverse covariance matrix.

This program takes "data" in the form of a fits image.

For best results data should take the form of a binary map where the image has the value float(1.0) for every pixel
that should be considered and float(0.0) where the pixel should not be considered. 
(This could be done for example by applying a sigma cut every value below which is set to zero, 
every value above which is set to one)

Thus the bayes fit only cares about the shape of the object in the image.

I would suggest you do look and understand the priors to constrain the ellipse better 
(currently they are set to constrain the central pixel to the middle 20% in x and y of the image,
i.e. I assume your object is a cut out with the object roughly in the center of the map.
This can be taken out and x0 and y0 can be found within any part of the map quite successfully.

### Example of running this code

```python
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
```
