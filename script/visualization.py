import numpy as np
from matplotlib.patches import Ellipse


def plot_sample_trajectories(subplot, abs_x, abs_y, start, abs_y_hat, line_args=None):
    if not line_args:
        line_args = dict()

    # plot observed and ground truth trajectory
    if line_args is None:
        line_args = {}
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), dim=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)

    # plot predicted trajectories(may sample many times)
    sample_times = abs_y_hat.shape[0]
    for t in range(sample_times):
        # all paths
        abs_y_hat_cat_x = np.concatenate((start.repeat(sample_times, axis=0), abs_y_hat), axis=1)
        if t == 0:
            subplot.plot(abs_y_hat_cat_x[t, :, 0], abs_y_hat_cat_x[t, :, 1], color='deeppink', label='y_hat',
                         **line_args)
        else:
            subplot.plot(abs_y_hat_cat_x[t, :, 0], abs_y_hat_cat_x[t, :, 1], color='deeppink', **line_args)
    return subplot


def plot_gaussian_ellipse(subplot, abs_x, abs_y, start, gaussian_output, confidence,
                          line_args=None, ellipse_args=None):
    if not line_args:
        line_args = dict()
    if not ellipse_args:
        ellipse_args = dict()

    # plot observed and ground truth trajectory
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), dim=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)

    # plot center of gaussian
    abs_y_hat = gaussian_output[:, :, 0:2]
    abs_y_hat_cat_x = np.concatenate((start, abs_y_hat), axis=1)
    subplot.plot(abs_y_hat_cat_x[0, :, 0], abs_y_hat_cat_x[0, :, 1], color='deeppink', label='y_hat', **line_args)

    # plot ellipse of gaussian
    seq_len = abs_y_hat.shape[1]
    for step in range(0, seq_len):
        mux, muy, sx, sy, rho = np.split(abs_y_hat[0, step, :], indices_or_sections=5)
        ellipse_patch = get_2d_gaussian_error_ellipse(mux=mux, muy=muy, sx=sx, sy=sy, rho=rho, confidence=confidence,
                                                      **ellipse_args)
        subplot.add_patch(ellipse_args)
    return subplot


def plot_potential_zone(subplot, abs_x, abs_y, start, gaussian_output, confidence, line_args=None, ellipse_args=None):
    pass


def get_2d_gaussian_error_ellipse(mux, muy, sx, sy, rho, confidence, **kwargs):
    """
    Return a patch of 2D gaussian distribution Ellipse.
    :return: cls matplotlib.patches
    """
    covmat = np.array([[sx * sx, sx * sy * rho], [sx * sy * rho, sy * sy]], dtype=float)
    eig_val, eigen_vec = np.linalg.eig(covmat)
    if eig_val[0] > eig_val[1]:
        major_eig_val, minor_eig_val = eig_val[0], eig_val[1]
        major_eig_vector, minor_eig_vector = eigen_vec[:, 0], eigen_vec[:, 1]
    else:
        major_eig_val, minor_eig_val = eig_val[1], eig_val[0]
        major_eig_vector, minor_eig_vector = eigen_vec[:, 1], eigen_vec[:, 0]
    ell_width = 2 * np.sqrt(confidence * major_eig_val)
    ell_height = 2 * np.sqrt(confidence * minor_eig_val)
    angle = np.ma.arctan(major_eig_vector[1] / major_eig_vector[0])
    angle = angle * 180 / np.pi
    error_ellipse = Ellipse((mux, muy), width=ell_width, height=ell_height, angle=angle, **kwargs)
    return error_ellipse
