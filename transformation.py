from collections import namedtuple
from itertools import product
from functools import reduce
import numpy as np
import xarray as xr


Point = namedtuple('Point', ('x', 'y'))


def bin_centers(bin_edges):
    return .5 * (bin_edges[1:] + bin_edges[:-1])


def get_aspect(image_data):
    '''Calculates aspect ratio of an image based on the (x, y) units
    
    Parameters:
    -----------
    image_data : xarray.DataArray or xarray.Dataset with 'x' and 'y' in dimensions
    
    Returns:
    --------
    aspect_ratio : float
    '''
    return (image_data.x[-1]-image_data.x[0]) / (image_data.y[-1]-image_data.y[0])


def cart_to_polar_map(x, y, origin):
    '''Mapping from Cartesian (x, y) to polar (theta, r) coordinates in an image
    
    Parameters
    ----------
    x : array-like
        x-coordinates of the image.
    y : array-like
        y-coordinates of the image.
    origin: namedtuple
        origin coordinates (xo, yo).
        
    Returns
    -------
    xarray.Dataset
        Dataset containing Cartesian and polar coordinates for each image pixel
    '''
    X, Y = np.meshgrid(x - origin.x, y - origin.y)
    return xr.Dataset({'theta': (('y', 'x'), np.arctan2(X, Y)), 
                       'r': (('y', 'x'), np.sqrt(X*X + Y*Y))}, 
                      coords={'y': y, 'x': x}, attrs={'origin': origin})


def make_bins(crd_map, *, theta=np.linspace(-np.pi, np.pi, 2), r=np.arange(0, 3., .5)): 
    ''' 2D binning of pixels in an image based on their polar (theta, r) coordinates.
    Returned object provides easy and direct visualisation of each bin on ('x', 'y') scale.
    
    Parameters
    ----------
    crd_map : xarray.Dataset
        output of 'cart_to_polar_map' function.
    theta : array-like
        bin edges for theta coordinate.
    r : array-like
        bin edges for r coordinate.
        
    Returns
    -------
    xarray.Dataset
        Dataset.data attribute contains 2d masks of each bin in (x, y)-plane corresponding to each
        (theta, r) coordinate.
        Dataset.n contains the number of pixels in each bin.
        
    '''
    bin_masks = np.stack(
        reduce(np.logical_and, (crd_map.r < r2, crd_map.r >= r1, crd_map.theta <= th2, crd_map.theta > th1))
        for (th1, th2), (r1, r2) in product(zip(theta, theta[1:]), zip(r, r[1:]))
                        )
    coords = (bin_centers(theta), bin_centers(r), crd_map.y, crd_map.x)
    binning = xr.DataArray(bin_masks.reshape([len(i) for i in coords]), 
                             coords=coords, dims=('theta', 'r', 'y', 'x'), name='binning')
    binning.coords['n'] = (('theta', 'r'), np.sum(bin_masks, axis=(1,2)).reshape(len(theta)-1, len(r)-1))
    return binning


def transform_to_polar(data, binning):
    ''' Transform image data into polar coordinates based on chosen binning. Pixels within each bin are
    averaged.
    
    Parameters:
    -----------
    data : xarray.DataArray or xarray.Dataset
        Image data. Should have dimensions ('t', 'y', 'x') for a time series or ('y', 'x') for a single image.
    
    binning : xarray.Dataset
        output of 'make_bins' function.
    
    Returns:
    --------
    polar_data : xarray.Dataset
        image data in polar coordinates. 
        polar_data.y contains the average pixels intensity in each (theta, r) bin
        polar_data.yerr contains the std for polar_data.y values
    '''
    def calc_stats(polar_bin):
        return xr.Dataset({'y': data.where(polar_bin, drop=True).mean(('y', 'x')), 
                           'yerr': data.where(polar_bin, drop=True).std(('y', 'x')) / np.sqrt(polar_bin.n)})
    return binning.groupby('theta').apply(lambda x: x.groupby('r').apply(calc_stats))


def plot_bins(binning, size=6):
    '''  Pixels binned together on ('x', 'y')-plane. Each bin contains multiple pixels 
    corresponding to (theta, r) point in polar coordinates.
    Neighbor bins are colored differently for display purpose only.
    
    Parameters:
    -----------
    binning : xarray.Dataset
        output of 'make_bins' function.
        
    size : float or int
        size of the figure.
    '''
    def alt_sign_array(ny, nx):
        row = np.stack((-1)**i for i in range(nx))
        return np.vstack((-1)**i * row for i in range(ny))
    
    alt_signs = xr.DataArray(alt_sign_array(*binning.shape[:2]), dims=('theta', 'r'))
    all_bins = (binning * alt_signs).sum(('theta', 'r'))
    all_bins.plot.imshow(size=size, aspect=get_aspect(all_bins), add_colorbar=0)
    
    
def circ_aperture(image_shape=(256 + 1, 256 + 1), origin=Point(128, 128), radius=25.):
    ''' Binary circular aperture '''
    nx, ny = image_shape
    radius_map = cart_to_polar_map(np.arange(nx), np.arange(ny), origin).r
    return radius_map <= radius


def radial_intensity_MSD(origin, image, theta_bins, r_bins):
    ''' Mean squared deviation of radial intensity profiles at different theta from the average profile (over 2pi) '''
    coord_map = cart_to_polar_map(image.x, image.y, Point(*origin))
    binning = make_bins(coord_map, theta=theta_bins, r=r_bins)
    polar = transform_to_polar(image, binning)
    residuals = (polar.y - polar.y.mean('theta'))**2
    return float(residuals.mean(('theta', 'r')))