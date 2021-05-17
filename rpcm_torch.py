"""
This script takes a RPC model and performs the localization function using torch
This way tensors can be used as input instead of numpy arrays
rpcm: https://github.com/cmla/rpcm
"""

import torch

class MaxLocalizationIterationsError(Exception):
    """
    Custom rpcm Exception.
    """
    pass

def apply_poly(poly, x, y, z):
    """
    Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.
    Args:
        poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
            ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.
    Returns:
        the value(s) of the polynom on the input point(s).
    """
    out = 0
    out += poly[0]
    out += poly[1]*y + poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out


def apply_rfm(num, den, x, y, z):
    """
    Evaluates a Rational Function Model (rfm), on a triplet of numbers.
    Args:
        num: list of the 20 coefficients of the numerator
        den: list of the 20 coefficients of the denominator
            All these coefficients are ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.
    Returns:
        the value(s) of the rfm on the input point(s).
    """
    return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)


def localization(rpc, col, row, alt, return_normalized=False):
    """
    Convert image coordinates plus altitude into geographic coordinates.
    Args:
        col (float or list): x image coordinate(s) of the input point(s)
        row (float or list): y image coordinate(s) of the input point(s)
        alt (float or list): altitude(s) of the input point(s)
    Returns:
        float or list: longitude(s)
        float or list: latitude(s)
    """
    ncol = (col - rpc.col_offset) / rpc.col_scale
    nrow = (row - rpc.row_offset) / rpc.row_scale
    nalt = (alt - rpc.alt_offset) / rpc.alt_scale

    if not hasattr(rpc, 'lat_num'):
        lon, lat = localization_iterative(rpc, ncol, nrow, nalt)
    else:
        lon = apply_rfm(rpc.lon_num, rpc.lon_den, nrow, ncol, nalt)
        lat = apply_rfm(rpc.lat_num, rpc.lat_den, nrow, ncol, nalt)

    if not return_normalized:
        lon = lon * rpc.lon_scale + rpc.lon_offset
        lat = lat * rpc.lat_scale + rpc.lat_offset

    return lon, lat


def localization_iterative(rpc, col, row, alt, delta=0.1):
    """
    Iterative estimation of the localization function (image to ground),
    for a list of image points expressed in image coordinates.
    Args:
        col, row: normalized image coordinates (between -1 and 1)
        alt: normalized altitude (between -1 and 1) of the corresponding 3D
            point
    Returns:
        lon, lat: normalized longitude and latitude
    Raises:
        MaxLocalizationIterationsError: if the while loop exceeds the max
            number of iterations, which is set to 100.
    """
    # target point: Xf (f for final)
    Xf = torch.vstack([col, row]).T

    # use 3 corners of the lon, lat domain and project them into the image
    # to get the first estimation of (lon, lat)
    # EPS is 2 for the first iteration, then 0.1.
    lon = -col ** 0 * delta # vector of ones
    lat = -col ** 0 * delta
    EPS = 2 * delta
    x0 = apply_rfm(rpc.col_num, rpc.col_den, lat, lon, alt)
    y0 = apply_rfm(rpc.row_num, rpc.row_den, lat, lon, alt)
    x1 = apply_rfm(rpc.col_num, rpc.col_den, lat, lon + EPS, alt)
    y1 = apply_rfm(rpc.row_num, rpc.row_den, lat, lon + EPS, alt)
    x2 = apply_rfm(rpc.col_num, rpc.col_den, lat + EPS, lon, alt)
    y2 = apply_rfm(rpc.row_num, rpc.row_den, lat + EPS, lon, alt)

    n = 0
    while not torch.all((x0 - col) ** 2 + (y0 - row) ** 2 < 1e-18):

        if n > 100:
            raise MaxLocalizationIterationsError("Max localization iterations (100) exceeded")

        X0 = torch.vstack([x0, y0]).T
        X1 = torch.vstack([x1, y1]).T
        X2 = torch.vstack([x2, y2]).T
        e1 = X1 - X0
        e2 = X2 - X0
        u  = Xf - X0

        # project u on the base (e1, e2): u = a1*e1 + a2*e2
        # the exact computation is given by:
        #   M = np.vstack((e1, e2)).T
        #   a = np.dot(np.linalg.inv(M), u)
        # but I don't know how to vectorize this.
        # Assuming that e1 and e2 are orthogonal, a1 is given by
        # <u, e1> / <e1, e1>
        num = torch.sum(torch.multiply(u, e1), axis=1)
        den = torch.sum(torch.multiply(e1, e1), axis=1)
        a1 = torch.divide(num, den).squeeze()

        num = torch.sum(torch.multiply(u, e2), axis=1)
        den = torch.sum(torch.multiply(e2, e2), axis=1)
        a2 = torch.divide(num, den).squeeze()

        # use the coefficients a1, a2 to compute an approximation of the
        # point on the gound which in turn will give us the new X0
        lon += a1 * EPS
        lat += a2 * EPS

        # update X0, X1 and X2
        EPS = .1
        x0 = apply_rfm(rpc.col_num, rpc.col_den, lat, lon, alt)
        y0 = apply_rfm(rpc.row_num, rpc.row_den, lat, lon, alt)
        x1 = apply_rfm(rpc.col_num, rpc.col_den, lat, lon + EPS, alt)
        y1 = apply_rfm(rpc.row_num, rpc.row_den, lat, lon + EPS, alt)
        x2 = apply_rfm(rpc.col_num, rpc.col_den, lat + EPS, lon, alt)
        y2 = apply_rfm(rpc.row_num, rpc.row_den, lat + EPS, lon, alt)

        n += 1

    return lon, lat
