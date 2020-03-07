import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.filters import gaussian_filter


def plot_sample_trajectories(subplot, abs_x, abs_y, start, abs_y_hat, line_args=None):
    if not line_args:
        line_args = dict()

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

    # plot observed and ground truth trajectory
    if line_args is None:
        line_args = {}
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), axis=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)

    # extents
    extents = get_extents(abs_x, abs_y, abs_y_hat)
    subplot.axis(extents)

    return subplot


def plot_gaussian_ellipse(subplot, abs_x, abs_y, start, gaussian_output, confidence,
                          line_args=None, ellipse_args=None):
    if not line_args:
        line_args = dict()
    if not ellipse_args:
        ellipse_args = dict()

    if gaussian_output.shape[0] > 1:
        print('Found multiple predicted gaussian output, only print the first.')

    # plot center of gaussian
    abs_y_hat = gaussian_output[:, :, 0:2]
    abs_y_hat_cat_x = np.concatenate((start, abs_y_hat), axis=1)
    subplot.plot(abs_y_hat_cat_x[0, :, 0], abs_y_hat_cat_x[0, :, 1], color='deeppink', label='y_hat', **line_args)
    subplot.scatter(abs_y_hat_cat_x[0, :, 0], abs_y_hat_cat_x[0, :, 1], color='blue', marker='o')

    # plot ellipse of gaussian
    seq_len = abs_y_hat.shape[1]
    for step in range(0, seq_len):
        mux, muy, sx, sy, rho = np.split(gaussian_output[0, step, :], indices_or_sections=5)
        ellipse_patch = get_2d_gaussian_error_ellipse(mux=mux[0], muy=muy[0], sx=sx[0], sy=sy[0], rho=rho[0],
                                                      confidence=confidence, **ellipse_args)
        subplot.add_patch(ellipse_patch)

    # plot observed and ground truth trajectory
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), axis=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)

    # extents
    extents = get_extents(abs_x, abs_y, gaussian_output[:, :, 0:2])
    subplot.axis(extents)

    return subplot


def plot_potential_zone(subplot, abs_x, abs_y, start, gaussian_output, confidence_zone, line_args=None,
                        patch_args=None, ellipse_args=None):
    # todo : absolute angle can not fit all situation, must find other way to check the bound.
    if not line_args:
        line_args = dict()
    if not patch_args:
        patch_args = dict()
    if not ellipse_args:
        ellipse_args = dict()
    if gaussian_output.shape[0] > 1:
        print('Found multiple predicted gaussian output, only print the first.')

    # plot observed and ground truth trajectory
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), axis=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)
    potential_fields = get_potential_zone(start=(start[0, 0, 0], start[0, 0, 1]),
                                          gaussian_output=gaussian_output[0, ...],
                                          confidence_zone=confidence_zone,
                                          **patch_args)
    for f in potential_fields:
        subplot.add_patch(f)

    # plot final ellipse
    mux, muy, sx, sy, rho = np.split(gaussian_output[0, -1, :], indices_or_sections=5)
    ellipse_patch = get_2d_gaussian_error_ellipse(mux=mux[0], muy=muy[0], sx=sx[0], sy=sy[0], rho=rho[0],
                                                  confidence=confidence_zone[1], **ellipse_args)
    subplot.add_patch(ellipse_patch)

    # extents
    extents = get_extents(abs_x, abs_y, gaussian_output[:, :, 0:2])
    subplot.axis(extents)
    return subplot


def plot_potential_heat_map(subplot, abs_x, abs_y, start, abs_y_hat, gaussian_output,
                            src='sample',
                            sample_times=50,
                            sigma=64,
                            bins=1000,
                            line_args=None):
    if not line_args:
        line_args = dict()

    # plot observed and ground truth trajectory
    subplot.plot(abs_x[0, :, 0], abs_x[0, :, 1], color='darkblue', label='x', **line_args)
    abs_y_cat_x = np.concatenate((start, abs_y), axis=1)
    subplot.plot(abs_y_cat_x[0, :, 0], abs_y_cat_x[0, :, 1], color='goldenrod', label='y_gt', **line_args)

    # get extents = heat map scale
    extents = get_extents(abs_x, abs_y, abs_y_hat)
    x_bin = np.arange(extents[0], extents[1], (extents[1] - extents[0]) / bins)
    y_bin = np.arange(extents[2], extents[3], (extents[3] - extents[2]) / bins)

    # Two types of source
    if src == 'sample':
        # [sample_times, pred_len, 2]
        xs = list()
        ys = list()
        lines_count = abs_y_hat.shape[0]
        for i in range(lines_count):
            x_begin, x_end = abs_y_hat[i, 0, 0], abs_y_hat[i, -1, 0]
            z = np.polyfit(abs_y_hat[i, :, 0], abs_y_hat[i, :, 1], 3)
            z = np.poly1d(z)
            xs += np.arange(x_begin, x_end, (x_end - x_begin) / 100).tolist()
            ys += np.polyval(z, np.arange(x_begin, x_end, (x_end - x_begin) / 100)).tolist()

        img, _ = plot_heat_map(xs, ys, sigma, bins=(x_bin, y_bin))
        subplot.imshow(img, extent=extents, origin='lower', cmap=plt.get_cmap('Greens'))
    elif src == 'gaussian_ellipse':
        pred_len = abs_y_hat.shape[1]
        xs = list()
        ys = list()
        for step in range(pred_len):
            g_slice = gaussian_output[0, step, :]
            mux, muy, sx, sy, rho = g_slice[0], g_slice[1], g_slice[2], g_slice[3], g_slice[4]
            mean = (mux, muy)
            cov = ((sx * sx, sx * sy * rho), (sx * sy * rho, sy * sy))
            samples = np.random.multivariate_normal(mean, cov, sample_times)
            xs += [samples[i, 0] for i in range(sample_times)]
            ys += [samples[i, 1] for i in range(sample_times)]
        img, _ = plot_heat_map(xs, ys, sigma=sigma, bins=(x_bin, y_bin))
        subplot.imshow(img, extent=extents, origin='lower', cmap=plt.get_cmap('Greens'))
    else:
        raise Exception('Heat map plot not supports {}'.format(src))

    subplot.axis(extents)
    subplot.axis('auto')
    return subplot


def get_2d_gaussian_error_ellipse(mux, muy, sx, sy, rho, confidence, **kwargs):
    """
    Return a patch of 2D gaussian distribution Ellipse.
    Confidence calculation: Chi-Squared distribution.
        (ref https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/)
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


def get_potential_zone(start, gaussian_output, confidence_zone, **kwargs):
    """
    Return a patch.path of potential field based on 2D gaussian distribution.
    :param start: the previous step of first prediction.
    :param gaussian_output: [seq_len, 5]
    :param confidence_zone: confidence value lower bound, tuple(lower, upper)
    :return: patch.path cls
    """
    paths = list()
    confidence_lower, confidence_upper = confidence_zone
    for log_confidence in np.arange(np.log(confidence_upper), np.log(confidence_lower), -0.1):
        confidence_zone = np.exp(log_confidence)
        verts = list()
        codes = list()
        verts.append(start)
        codes.append(Path.MOVETO)
        seq_len = gaussian_output.shape[0]
        # one side angle = 0
        for step in range(0, seq_len):
            mux, muy, sx, sy, rho = gaussian_output[step, :].tolist()
            vert = get_vert_of_error_ellipse_by_angle((mux, muy, sx, sy, rho), confidence_zone, angle=-90)
            verts.append(vert)

        # curve of the semi-ellipse of gaussian distribution in the last step
        for angle in range(-90, 90, 10):
            mux, muy, sx, sy, rho = gaussian_output[seq_len - 1, :].tolist()
            vert = get_vert_of_error_ellipse_by_angle((mux, muy, sx, sy, rho), confidence_zone, angle=angle)
            verts.append(vert)

        # one side angle = 180
        for step in range(seq_len - 1, -1, -1):
            mux, muy, sx, sy, rho = gaussian_output[step, :].tolist()
            vert = get_vert_of_error_ellipse_by_angle((mux, muy, sx, sy, rho), confidence_zone, angle=90)
            verts.append(vert)

        for i in range(1, len(verts)):
            codes.append(Path.CURVE3)

        verts.append(start)
        codes.append(Path.CLOSEPOLY)

        path = Path(vertices=verts, codes=codes)
        patch_path = patches.PathPatch(path, lw=0, **kwargs)
        paths.append(patch_path)

    return paths


def get_vert_of_error_ellipse_by_angle(gaussian_parameters, confidence, angle):
    """
    Return a (x,y) 2D point at the error ellipse of gaussian distribution.
    :param gaussian_parameters: (mux muy, sx, sy, rho)
    :param confidence: to decide how large the error ellipse is.
    :param angle: shift from the major axis of error ellipse.
    :return: (x,y)

    Theory:
        1. Get transformation A matrix from given gaussian_parameters.
        2. Define norm vert by angle in norm gaussian distribution.
        3. Use A to transform norm vert.
        4. Shift with mux and muy.
        ref (Chinese): https://zhuanlan.zhihu.com/p/37609917
    """
    # define vert with angle in norm 2D gaussian distribution.
    angle = angle * np.pi / 180
    norm_vert = np.zeros((2, 1))
    norm_vert[0, 0], norm_vert[1, 0] = np.sqrt(confidence) * np.ma.cos(angle), np.sqrt(confidence) * np.ma.sin(
        angle)
    mux, muy, sx, sy, rho = gaussian_parameters

    # get eig value and vector
    covmat = np.array([[sx * sx, sx * sy * rho], [sx * sy * rho, sy * sy]])
    eig_val, eig_vec = np.linalg.eig(covmat)
    eig_val = np.array([[eig_val[0], 0], [0, eig_val[1]]])

    # get transform
    transform_mat = np.dot(eig_vec, np.sqrt(eig_val))
    transformed_vert = np.dot(transform_mat, norm_vert)

    # shift by mux, muy
    transformed_vert[0] += mux
    transformed_vert[1] += muy

    return transformed_vert[0], transformed_vert[1]


def plot_heat_map(x, y, sigma, bins):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def get_extents(x, y, y_hat):
    x_min = min(np.min(y_hat[:, :, 0]), np.min(y[:, :, 0]), np.min(x[:, :, 0]))
    x_max = max(np.max(y_hat[:, :, 0]), np.max(y[:, :, 0]), np.max(x[:, :, 0]))
    y_min = min(np.min(y_hat[:, :, 1]), np.min(y[:, :, 1]), np.min(x[:, :, 1]))
    y_max = max(np.max(y_hat[:, :, 1]), np.max(y[:, :, 1]), np.max(x[:, :, 1]))
    x_dif = (x_max - x_min) * 0.5
    y_dif = (y_max - y_min) * 0.5
    extents = [x_min - x_dif, x_max + x_dif, y_min - y_dif, y_max + y_dif]
    return extents


if __name__ == '__main__':

    colors = [
        '#fff700', '#fef400', '#fef200', '#fef000', '#feee00', '#feec00', '#fee900',
        '#fee700', '#fde500', '#fde300', '#fde100', '#fddf00', '#fddc00', '#fdda00', '#fdd800',
        '#fdd600', '#fcd400', '#fcd200', '#fccf00', '#fccd00', '#fccb00', '#fcc900', '#fcc700',
        '#fcc400', '#fbc200', '#fbc000', '#fbbe00', '#fbbc00', '#fbba00', '#fbb700', '#fbb500',
        '#fbb300', '#fab100', '#faaf00', '#faad00', '#faaa00', '#faa800', '#faa600', '#faa400',
        '#faa200', '#f9a000', '#f99d00', '#f99b00', '#f99900', '#f99700', '#f99500', '#f99200',
        '#f99000', '#f88e00', '#f88c00', '#f88a00', '#f88800', '#f88500', '#f88300', '#f88100',
        '#f87f00', '#f77d00', '#f77b00', '#f77800', '#f77600', '#f77400', '#f77200', '#f77000',
        '#f76e01']

    start = (100, 100)
    gaussian = [[100, 100, 2, 3, 0], [110, 110, 2, 4, 0], [120, 120, 2, 5, 0], [130, 130, 2, 6, 0]]
    gaussian = np.array(gaussian, dtype=float)

    fig, subplot = plt.subplots()

    fig.suptitle('hello world')

    potential_fields = get_potential_zone(start, gaussian, confidence_zone=(0.5, 5.99), colors='#65be99',
                                          alpha=0.1)

    for f in potential_fields:
        subplot.add_patch(f)

    plt.axis('scaled')
    plt.axis('equal')
    plt.show()
