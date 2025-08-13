"""Weather Forecast"""
import os
import math
import logging
import numpy as np
import torch

# from mindspore import amp

# from ..data import FEATURE_DICT, SIZE_DICT
from utils.data_loader_npyfiles import get_data_loader_npy, FEATURE_DICT, SIZE_DICT

PI = math.pi


class WeatherForecast:

    def __init__(self,
                 model,
                 config,
                 dataset,
                 run_mode='test',
                #  logger
                 ):
        # self.model = amp.auto_mixed_precision(model, config['train']['amp_level'])
        self.model = model
        # self.logger = logger
        self.config = config
        self.dataset = dataset
        self.device = next(self.model.parameters()).device
        self.adjust_size = False
        self.h_size, self.w_size = SIZE_DICT[config['grid_resolution']]
        if self.config['patch']:
            patch_size = [config['patch_size']]
            self.h_size = self.h_size - self.h_size % patch_size[0]
            self.adjust_size = True

        self.mean = self.dataset.mean_pressure_level
        self.mean_s = self.dataset.mean_surface
        self.total_mean = np.concatenate([self.mean.transpose((1,0)).reshape([-1]), self.mean_s], axis=-1)

        self.std = self.dataset.std_pressure_level
        self.std_s = self.dataset.std_surface
        self.total_std = np.concatenate([self.std.transpose((1,0)).reshape([-1]), self.std_s], axis=-1)

        # self.total_std = self._get_total_sample_description(config, "std")
        # self.total_mean = self._get_total_sample_description(config, "mean")
        # self.climate_mean = self._get_history_climate_mean(config, self.w_size, self.adjust_size)
        self.run_mode = run_mode
        if self.run_mode == 'test':
            self.t_out_test = self.config["t_out_test"]
        elif self.run_mode == 'valid':
            self.t_out_test = self.config["t_out_valid"]
        self.pred_lead_time = config['pred_lead_time']

    @staticmethod
    def _get_total_sample_description(config, info_mode):
        """get total sample std or mean description."""
        root_dir = config['root_dir']
        sample_info_pressure_levels = np.load(
            os.path.join(root_dir, "statistic", info_mode + ".npy"))
        sample_info_pressure_levels = sample_info_pressure_levels.transpose(1, 2, 3, 0)
        sample_info_pressure_levels = sample_info_pressure_levels.reshape((1, -1))
        sample_info_pressure_levels = np.squeeze(sample_info_pressure_levels, axis=0)
        sample_info_surface = np.load(os.path.join(root_dir, "statistic",
                                                   info_mode + "_s.npy"))
        total_sample_info = np.append(sample_info_pressure_levels, sample_info_surface)

        return total_sample_info

    @staticmethod
    def _get_history_climate_mean(config, w_size, adjust_size=False):
        """get history climate mean."""
        data_params = config['root_dir']
        climate_mean = np.load(os.path.join(config['root_dir'], "statistic",
                                            f"climate_{config['grid_resolution']}.npy"))
        feature_dims = climate_mean.shape[-1]
        if adjust_size:
            climate_mean = climate_mean.reshape(-1, w_size, feature_dims)[:-1].reshape(-1, feature_dims)

        return climate_mean

    def _get_absolute_idx(self, idx):
        return idx[1] * self.config['pressure_level_num'] + idx[0]

    def _print_key_metrics(self, rmse, acc):
        """print key info metrics"""
        # z500_idx = self._get_absolute_idx(FEATURE_DICT.get("Z500"))
        # t2m_idx = self._get_absolute_idx(FEATURE_DICT.get("T2M"))
        # t850_idx = self._get_absolute_idx(FEATURE_DICT.get("T850"))
        # u10_idx = self._get_absolute_idx(FEATURE_DICT.get("U10"))

        z500_idx = self._get_absolute_idx(FEATURE_DICT.get("Z500"))
        q500_idx = self._get_absolute_idx(FEATURE_DICT.get("Q500"))
        u500_idx = self._get_absolute_idx(FEATURE_DICT.get("U500"))
        v500_idx = self._get_absolute_idx(FEATURE_DICT.get("V500"))
        t500_idx = self._get_absolute_idx(FEATURE_DICT.get("T500"))
        t850_idx = self._get_absolute_idx(FEATURE_DICT.get("T850"))

        # u10_idx = self._get_absolute_idx(FEATURE_DICT.get("U10"))
        # v10_idx = self._get_absolute_idx(FEATURE_DICT.get("V10"))

        for timestep_idx in range(self.t_out_test):
            logging.info(f't = {self.pred_lead_time*(timestep_idx+1)} hour: ')

            # z500_rmse = rmse[z500_idx, timestep_idx]
            # z500_acc = acc[z500_idx, timestep_idx]
            # t2m_rmse = rmse[t2m_idx, timestep_idx]
            # t2m_acc = acc[t2m_idx, timestep_idx]
            # t850_rmse = rmse[t850_idx, timestep_idx]
            # t850_acc = acc[t850_idx, timestep_idx]
            # u10_rmse = rmse[u10_idx, timestep_idx]
            # u10_acc = acc[u10_idx, timestep_idx]
            # logging.info(f" RMSE of Z500: {z500_rmse}, T2m: {t2m_rmse}, T850: {t850_rmse}, U10: {u10_rmse}")
            # logging.info(f" ACC  of Z500: {z500_acc}, T2m: {t2m_acc}, T850: {t850_acc}, U10: {u10_acc}")

            z500_rmse = rmse[z500_idx, timestep_idx]
            q500_rmse = rmse[q500_idx, timestep_idx]

            u500_rmse = rmse[u500_idx, timestep_idx]
            v500_rmse = rmse[v500_idx, timestep_idx]

            t500_rmse = rmse[t500_idx, timestep_idx]
            t850_rmse = rmse[t850_idx, timestep_idx]

            # u10_rmse = rmse[u10_idx, timestep_idx]
            # v10_rmse = rmse[v10_idx, timestep_idx]

            logging.info(f" RMSE of Z500: {z500_rmse}, Q500: {q500_rmse}, U500: {u500_rmse}, V500: {v500_rmse}, T500: {t500_rmse}, T850: {t850_rmse}")

            # logging.info(f" RMSE of Z500: {z500_rmse}, Q500: {q500_rmse}, U500: {u500_rmse}, U850: {u850_rmse}, U10: {u10_rmse}, V10: {v10_rmse}")
            # logging.info(f" ACC  of Z500: {z500_acc}, T2m: {t2m_acc}, T850: {t850_acc}, U10: {u10_acc}")


    @staticmethod
    def forecast(inputs, labels=None):
        """
        The forecast function of the model.

        Args:
            inputs (Tensor): The input data of model.
            labels (Tensor): True values of the samples.
        """
        raise NotImplementedError("forecast module not implemented")

    def eval(self, data_loader):
        '''
        Eval the model using test dataset or validation dataset.

        Args:
            dataset (mindspore.dataset): The dataset for eval, including inputs and labels.
        '''
        logging.info("================================Start Evaluation================================")

        data_length = 0
        lat_weight_rmse = []
        lat_weight_acc = []
        # for data in dataset.create_dict_iterator():
        for i, data in enumerate(data_loader):
            # inputs = data['inputs']
            # labels = data['labels']
            inputs, labels, climates = data
            self.climates = climates
            # inputs, labels = data

            inputs = inputs.to(self.device, dtype = torch.float)
            batch_size = inputs.shape[0]
            
            lat_weight_rmse_step, lat_weight_acc_step = self._get_metrics(inputs, labels)
            if data_length == 0:
                lat_weight_rmse = lat_weight_rmse_step
                lat_weight_acc = lat_weight_acc_step
            else:
                lat_weight_rmse += lat_weight_rmse_step
                lat_weight_acc += lat_weight_acc_step

            data_length += batch_size

        logging.info(f'test dataset size: {data_length}')
        # (69, 20)
        lat_weight_rmse = np.sqrt(
            lat_weight_rmse / (data_length * self.w_size * self.h_size))
        lat_weight_acc = lat_weight_acc / data_length
        temp_rmse = lat_weight_rmse.transpose(1, 0)
        denormalized_lat_weight_rmse = temp_rmse * self.total_std
        denormalized_lat_weight_rmse = denormalized_lat_weight_rmse.transpose(1, 0)
        if self.config["save_rmse_acc"]:
            np.save(os.path.join(self.config['experiment_dir'],
                                 "denormalized_lat_weight_rmse.npy"), denormalized_lat_weight_rmse)
            np.save(os.path.join(self.config['experiment_dir'],
                                 "lat_weight_acc.npy"), lat_weight_acc)
        self._print_key_metrics(denormalized_lat_weight_rmse, lat_weight_acc)

        logging.info("================================End Evaluation================================")
        return denormalized_lat_weight_rmse, lat_weight_acc

    def _get_metrics(self, inputs, labels):
        """get metrics for plot"""
        pred = self.forecast(inputs, labels)
        feature_num = labels.shape[1]
        lat_weight_rmse = np.zeros((feature_num, self.t_out_test))
        lat_weight_acc = np.zeros((feature_num, self.t_out_test))
        for t in range(self.t_out_test):
            for f in range(feature_num):
                lat_weight_rmse[f, t] = self._calculate_lat_weighted_rmse(
                    labels[:, f, t].asnumpy(), pred[t][:, f].asnumpy())  # label(B,C,T,H W) pred(B,C,H W)
                lat_weight_acc[f, t] = self._calculate_lat_weighted_acc(
                    labels[:, f, t].asnumpy(), pred[t][:, f].asnumpy())
        return lat_weight_rmse, lat_weight_acc

    def _lat(self, j):
        return 90. - j * 180. / float(self.h_size - 1)

    def _latitude_weighting_factor(self, j, s):
        return self.h_size * np.cos(PI / 180. * self._lat(j)) / s

    def _calculate_lat_weighted_rmse(self, label, prediction):
        batch_size = label.shape[0]
        lat_t = np.arange(0, self.h_size)

        s = np.sum(np.cos(PI / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1)
        error = np.square(np.reshape(label, (batch_size, -1)) - np.reshape(prediction, (batch_size, -1)))
        lat_weight_error = np.sum(error * grid_node_weight)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """ calculate lat weighted acc"""
        lat_t = np.arange(0, self.h_size)

        s = np.sum(np.cos(PI / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s).reshape(self.h_size, 1)
        grid_node_weight = np.repeat(weight, self.w_size, axis=1)

        pred_prime = prediction
        label_prime = label
        a = np.sum(pred_prime * label * grid_node_weight)
        b = np.sqrt(np.sum(pred_prime ** 2 * grid_node_weight) * np.sum(label_prime ** 2 * grid_node_weight))
        acc = a / b
        return acc