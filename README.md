# bayes_single_ellipse_fitter
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
