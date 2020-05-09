import numpy as np
import torch
import os
from tqdm import tqdm

# from data.dataloader import KittiDataLoader
from data.kitti_dataloader import SingleKittiDataLoader
from model.utils import l2_loss

# ----
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from scipy.stats import multivariate_normal

is_show = False
log_likelihoods = []


def makedir(saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)


def pos_vel_filter_4_2(x, P, R, Q=0., dt=1.0):
    """
    Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x[0], x[1], x[2], x[3]])  # location and velocity: x,y,dx,dy
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement function: location

    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P
    if np.isscalar(Q):
        #        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        kf.Q *= Q
    else:
        kf.Q = Q

    if np.isscalar(R):
        kf.R *= R  # measurement uncertainty
    else:
        kf.R[:] = R

    return kf


def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.x = np.array([x[0], x[1], x[2], x[3], 0, 0, 0, 0])  # x,y,w,h,vx,vy,vw,vh
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0]])  # Measurement function: x,y,w,h

    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P
    if np.isscalar(Q):
        #        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        kf.Q *= Q
    else:
        kf.Q = Q

    if np.isscalar(R):
        kf.R *= R  # measurement uncertainty
    else:
        kf.R[:] = R

    return kf


def m_step(x_predictions, x_posteriors, measurements, H):
    """
    Input:
    - x_predictions: The predicted state variables at every time step
    - x_posteriors: The posterior state variables at every time step
        after performing the Kalman filter update (with the
        corresponding measurement).
    - measurements: The measurements at every time step.
    * note: Time step indexing is consistent between x_predictions,
        x_posteriors, and measurements
    -H: Measurement function matrix (Hx = x' where x is in the
        state space and x' is in the measurement space)

    Return:
    - Q: Maximum likelihood estimate of the process noise covariance
        matrix.  This is calculated as the covariance matrix of
        (x_posteriors - x_predictions)
    - R: Maximum likelihood estimate of the measurement noise covariance
        matrix.  This is calculated as the covariance matrix of
        (measurements - x_posteriors)
    * note: We would hope the mean of both (x_posteriors - x_predictions)
        and (measurements - x_posteriors) is zero, but this is not
        guaranteed.  Think about adjusting motion model and measurement
        function to account for this.
    """

    Q = np.cov((x_posteriors - x_predictions).T, bias=True)
    R = np.cov((measurements.T - np.dot(H, x_posteriors.T)), bias=True)
    return (Q, R)


def calc_LL(x_posteriors, cov, measurements, H, R):
    assert (len(x_posteriors) == len(cov) and len(cov) == len(measurements))
    LL = 0
    #    print("len(cov) = ", len(cov))
    #    print("cov.shape = ", cov.shape)
    for i in range(0, len(cov)):
        #        print("shape H = ", H.shape)
        #        print("shape P[i] = ", cov[i].shape)
        #        print("shape R = ", R.shape)
        #  Extract covariance matrix:   cov = ((sx * sx, rho * sx * sy), (rho * sx * sy, sy * sy))
        S = np.dot(np.dot(H, cov[i]),
                   H.T) + R  # HPH' + R  project system uncertainty into measurement space  [[19.95024876  0.        ], [ 0.         19.9950025 ]]
        # Extract mean: mean = (mux, muy)
        state_posterior_in_meas_space = np.dot(H, x_posteriors[i])  # Hx  [ 2.54367572 23.49578423]
        # distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S)
        distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S, allow_singular=True)  # --xh
        LL += np.log(distribution.pdf(measurements[i]))
    return LL


def calc_NLL(x_next_prediction, cov_next_prediction, gt, H, R):
    """
    calculate negative log likelihood
    x_posterior, cov: correction in t
    measurement: gt in t+1
    """
    # assert (len(x_next_prediction) == len(cov_next_prediction) and len(cov_next_prediction) == len(gt))
    #  Extract covariance matrix:   cov = ((sx * sx, rho * sx * sy), (rho * sx * sy, sy * sy))  (2, 2)
    # size 2*4 x 4*4 x 4*2 + 2*2 = 2*2
    S = np.dot(np.dot(H, cov_next_prediction),
               H.T) + R  # HPH' + R  project system uncertainty into measurement space  [cov]
    # Extract mean: mean = (mux, muy)
    state_posterior_in_meas_space = np.dot(H, x_next_prediction)  # Hx   [mean]
    # distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S)
    distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S, allow_singular=True)  # --xh
    NLL = -np.log(distribution.pdf(gt))
    if NLL == float("inf"):
        print('--------inf--------')
    return np.array(NLL)


def run_EM_on_Q_R(x0=(0., 0., 0., 0.), P=500, R=.0, Q=.0, dt=1.0, data=None, gt=None, obs_len=3):
    # def run_EM_on_Q_R(x0=(0., 0., 0., 0., 0., 0., 0., 0.), P=500, R=0, Q=0, dt=1.0, data=None):
    # log_likelihoods = []
    for i in range(0, obs_len):
        # print("iteration ", i, ":")
        # print("Q = ", Q)
        # print("R = ", R)

        # try:
        #     x_posteriors, x_predictions, x_next_prediction, cov, H, R = run(x0=x0, P=P, R=R, Q=Q, dt=dt, data=data)  # -> run()
        # except:
        #     print('---exception----')
        x_posteriors, x_predictions, x_next_prediction, cov_next_prediction, covs, H, R = run(x0=x0, P=P, R=R, Q=Q,
                                                                                              dt=dt,
                                                                                              data=data)  # -> run()

        # log_likelihoods.append(calc_LL(x_posteriors, cov, data, H, R))  # -> calc_LL()
        log_likelihood = calc_NLL(x_next_prediction, cov_next_prediction, gt, H, R)  # -> calc_NLL()
        Q, R = m_step(x_predictions, x_posteriors, data, H)
    return x_posteriors, x_predictions, x_next_prediction, log_likelihood  # posteriors, priors, predictors


def get_cmap(N):
    """
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def run(x0=(0., 0., 0., 0.), P=500, R=0, Q=0, dt=1.0, data=None):
    """
    `data` is a (number_of_measurements x dimension_of_measurements)
    numpy array containing the measurements
    """

    # create the Kalman filter
    kf = pos_vel_filter_4_2(x=x0, R=R, P=P, Q=Q, dt=1)
    # kf = pos_vel_filter(x=x0, R=R, P=P, Q=Q, dt=1)

    # run the kalman filter and store the results
    x_posteriors, x_predictions, cov = [], [], []
    for z in data:  # use history k-1 to predict k -> use measurement k to update
        kf.predict()  # Time update (prediction)
        x_predictions.append(kf.x)
        kf.update(z)  # Measurement update (correction)
        x_posteriors.append(kf.x)
        cov.append(kf.P)  # error covariance
    kf.predict()  # add by xh, for next prediction k+1
    # x_posteriors, x_predictions, cov = np.array(x_posteriors), np.array(x_predictions), np.array(cov)
    x_posteriors, x_predictions, x_next_prediction, cov_next_prediction, covs = \
        np.array(x_posteriors), np.array(x_predictions), np.array(kf.x), np.array(kf.P), np.array(cov)

    # print("cov.shape = ", covs.shape)

    if is_show:
        cmap = get_cmap(100)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_posteriors[:, 0], x_posteriors[:, 1], marker='+', c=cmap(1))  # x_posteriors: (x,y)
        ax.scatter(data[:, 0], data[:, 1], marker='*', c=cmap(99))
        plt.show()

    return x_posteriors, x_predictions, x_next_prediction, cov_next_prediction, covs, kf.H, kf.R


def test_data_splitter(batch_data, pred_len):
    """
    Split data [batch_size, total_len, 2] into datax and datay in train mode
    :param batch_data: data to be split
    :param pred_len: length of trajectories in final loss calculation
    :return: datax, datay
    """
    return batch_data[:, :-pred_len, :], batch_data[:, -pred_len:, :]


class KF:
    def __init__(self, args, recorder):
        print("--KF init--")
        self.args = args
        self.recorder = recorder
        self.device = torch.device('cpu')
        # old data loader.
        # self.test_dataset = KittiDataLoader(self.args.test_dataset,
        #                                     1,  # batch_size
        #                                     self.args.obs_len + self.args.pred_len,
        #                                     self.device)
        self.test_dataset = SingleKittiDataLoader(file_path=self.args.test_dataset,
                                                  batch_size=1,
                                                  trajectory_length=self.args.obs_len + self.args.pred_len,
                                                  device=self.device,
                                                  train_leave=None,
                                                  valid_scene=None)
        # self.predict()
        self.args_check()

    def args_check(self):
        if not self.args.use_sample and self.args.sample_times > 1:
            self.recorder.logger.info('Found not using sample, but sample times > 1. Auto turned to 1.')
            self.args.sample_times = 1

    def predict_then_evaluate(self, step=1):
        self.recorder.logger.info('### Begin Evaluation {}, {} test cases in total'.format(
            step, len(self.test_dataset))
        )
        save_list = list()
        process = tqdm(range(len(self.test_dataset)))

        # P = np.diag([10., 1., 10., 1.]) * 1000
        # P = np.diag([5, 10., 50, 100.])  # x,y,dx,dy
        P = np.diag([5, 10., 5, 10.])  # x,y,dx,dy
        for t in range(len(self.test_dataset)):
            process.update(n=1)
            batch = self.test_dataset.next_batch()
            data, rel_data = batch['data'], batch['rel_data']
            # split to x->y
            x, y = test_data_splitter(data, self.args.pred_len)
            xdata = x.detach().numpy()[0]
            ydata = y.detach().numpy()[0]
            x_posteriors, x_predictions, x_next_prediction, loss_nll = run_EM_on_Q_R(R=10, Q=.01, P=P, data=xdata,
                                                                                     gt=ydata,
                                                                                     obs_len=self.args.obs_len)
            # [[ 2.556394 23.507532], [ 2.622816 23.351763], [ 2.689237 23.195993]]
            # print(log_likelihoods)

            # numpy to Tensor
            # loss_nll = torch.tensor(loss_nll).to(self.device)  # float -> tensor
            loss_nll = torch.from_numpy(loss_nll).type(torch.float).to(self.device)
            y_hat = torch.from_numpy(x_next_prediction[:2]).type(torch.float).to(self.device)

            abs_x = x.detach().clone()
            abs_y = y.detach().clone()
            abs_y_hat = y_hat.detach().clone()

            # norm to raw
            abs_x = self.test_dataset.norm_to_raw(abs_x)
            abs_y = self.test_dataset.norm_to_raw(abs_y)
            abs_y_hat = self.test_dataset.norm_to_raw(abs_y_hat)

            # metric calculate
            loss = loss_nll  # norm scale
            l2 = l2_loss(y_hat, y)  # norm scale
            euler = l2_loss(abs_y_hat, abs_y)  # raw scale

            # average metrics calculation
            # Hint: when mode is absolute, abs_? and ? are the same, so L2 loss and destination error as well.
            ave_loss = torch.sum(loss) / (self.args.pred_len * self.args.sample_times)
            if len(loss.shape) != 0:
                first_loss = torch.sum(loss[:, 0, :]) / self.args.sample_times
                final_loss = torch.sum(loss[:, -1, :]) / self.args.sample_times
            else:
                first_loss = torch.sum(loss) / self.args.sample_times
                final_loss = torch.sum(loss) / self.args.sample_times

            ave_l2 = torch.sum(l2) / (self.args.pred_len * self.args.sample_times)
            final_l2 = torch.sum(l2[:, -1, :]) / self.args.sample_times

            ade = torch.sum(euler) / (self.args.pred_len * self.args.sample_times)
            fde = torch.sum(euler[:, -1, :]) / self.args.sample_times

            min_ade = torch.min(torch.sum(euler, dim=[1, 2]) / self.args.pred_len)
            min_fde = torch.min(euler[:, -1, :])

            msg1 = '{}_AveLoss_{:.3}_AveL2_{:.3}_FinalL2_{:.3}'.format(
                t, ave_loss, ave_l2, final_l2)
            msg2 = '{}_Ade_{:.3}_Fde_{:.3}_MAde_{:.3f}_MFde_{:.3f}'.format(
                t, ade, fde, min_ade, min_fde)
            if not self.args.silence:
                self.recorder.logger.info(msg1 + "_" + msg2)

            # plot
            record = dict()
            record['tag'] = t
            record['step'] = step
            record['title'] = msg2

            record['x'] = x.cpu().numpy()
            record['abs_x'] = abs_x.cpu().numpy()
            record['y'] = y.cpu().numpy()
            record['abs_y'] = abs_y.cpu().numpy()
            record['y_hat'] = y_hat.cpu().numpy()
            record['abs_y_hat'] = abs_y_hat.cpu().numpy()
            # record['gaussian_output'] = pred_gaussian.cpu().numpy()

            record['ave_loss'] = ave_loss.cpu().numpy()
            record['final_loss'] = final_loss.cpu().numpy()
            record['first_loss'] = first_loss.cpu().numpy()
            record['ave_l2'] = ave_l2.cpu().numpy()
            record['final_l2'] = final_l2.cpu().numpy()
            record['ade'] = ade.cpu().numpy()
            record['fde'] = fde.cpu().numpy()
            record['min_ade'] = min_ade.cpu().numpy()
            record['min_fde'] = min_fde.cpu().numpy()

            save_list.append(record)

        process.close()

        # globally average metrics calculation
        self.recorder.logger.info('Calculation of Global Metrics.')
        metric_list = ['ave_loss', 'final_loss', 'first_loss', 'ave_l2', 'final_l2', 'ade', 'fde', 'min_ade', 'min_fde']
        scalars = dict()
        for metric in metric_list:
            temp = list()
            for record in save_list:
                temp.append(record[metric])
            self.recorder.logger.info('{} : {}'.format(metric, sum(temp) / len(temp)))
            scalars[metric] = sum(temp) / len(temp)
            self.recorder.writer.add_scalar('{}_{}_Eval/{}'.format(self.args.model, self.args.phase, metric),
                                            scalars[metric], global_step=step)

        # plot
        if self.args.plot:
            self.recorder.logger.info('Plot trajectory')
            self.recorder.plot_trajectory(save_list, step=step, cat_point=self.args.obs_len - 1,
                                          mode=self.args.plot_mode)

        # export
        if self.args.export_path:
            torch.save(save_list, self.args.export_path)
            self.recorder.logger.info('Export {} Done'.format(self.args.export_path))

        self.recorder.logger.info('### End Evaluation')
