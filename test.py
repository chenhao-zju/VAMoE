import os
import time
import copy
import argparse
import numpy as np
import torch
import math
import logging
from utils import logging_utils
from utils.YParams import YParams

from utils.data_loader_npyfiles import get_data_loader_npy, FEATURE_DICT, SIZE_DICT, surface_features, higher_features, pressure_level
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels
# from model import SphericalFourierNeuralOperatorNet as SKNO
from networks import VAMoE
from inference import WeatherForecast

logging_utils.config_logger()

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    # try:
    new_state_dict = {}
    for key, val in checkpoint['model_state'].items():
        name = key[7:]
        # if name != 'ged':
        new_state_dict[name] = val  
    model.load_state_dict(new_state_dict)
    # except:
    #     model.load_state_dict(checkpoint['model_state'])
    # model.eval()
    return model

class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, dataset, run_mode='test', device='cpu'):
        super(InferenceModule, self).__init__(model, config, dataset, run_mode)
        # statistic_dir = os.path.join(config["data"]["root_dir"], "statistic")
        # mean = np.load(os.path.join(statistic_dir, "mean.npy"))
        # mean = mean.transpose(1, 2, 3, 0)  # HWFL(1, 1, 5, 13)
        # mean = mean.reshape((1, -1))
        # mean = np.squeeze(mean, axis=0)
        # mean_s = np.load(os.path.join(statistic_dir, "mean_s.npy"))
        # self.mean_all = np.concatenate([mean, mean_s], axis=-1)

        # std = np.load(os.path.join(statistic_dir, "std.npy"))
        # std = std.transpose(1, 2, 3, 0)
        # std = std.reshape((1, -1))
        # self.std = np.squeeze(std, axis=0)
        # self.std_s = np.load(os.path.join(statistic_dir, "std_s.npy"))
        # self.std_all = np.concatenate([self.std, self.std_s], axis=-1)
        self.dataset = dataset
        self.device = device

        self.mean = self.dataset.mean_pressure_level
        self.mean_s = self.dataset.mean_surface
        self.mean_all = np.concatenate([self.mean.transpose((1,0)).reshape([-1]), self.mean_s], axis=-1)

        self.std = self.dataset.std_pressure_level
        self.std_s = self.dataset.std_surface
        self.std_all = np.concatenate([self.std.transpose((1,0)).reshape([-1]), self.std_s], axis=-1)
        
        self.feature_dims = config['feature_dims']
        self.use_moe = config['use_moe']

        if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
            self.posembed = self.get_position()
            self.posembed = self.posembed.to(self.device, dtype = torch.float)

        self.climate = np.load("./climate_1.4_new.npy")

    def forecast(self, inputs, labels):
        rmse_lst = []
        acc_lst = []

        self.model.eval()
        with torch.no_grad():
            for i, t in enumerate( range(self.t_out_test) ):
                if self.config['loss'] == 'trainl2':
                    if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        pred, _, _ = self.model(inputs, target=labels[:, t], posembed=self.posembed)
                    else:
                        pred, _, _ = self.model(inputs, target=labels[:, t])
                else:                 
                    if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        pred, _ = self.model(inputs, posembed=self.posembed)
                    elif self.use_moe == 'moe':
                        pred, _, _ = self.model(inputs)
                    else:
                        pred, _ = self.model(inputs)    # .to(self.device, dtype = torch.float)

                rmse = weighted_rmse_torch_channels(pred, labels[:, t])
                acc = weighted_acc_torch_channels(pred, labels[:, t])

                rmse_lst.append(rmse)
                acc_lst.append(acc)

                inputs = pred


        total_rmse = torch.sum(torch.stack(rmse_lst, dim=-1), dim=0)
        total_acc = torch.sum(torch.stack(acc_lst, dim=-1), dim=0)
        return total_rmse, total_acc

    
    def eval(self, data_loader):
        '''
        Eval the model using test dataset or validation dataset.

        Args:
            dataset (mindspore.dataset): The dataset for eval, including inputs and labels.
        '''
        logging.info("================================Start Evaluation================================")

        data_length = 0
        lat_weight_rmse = torch.zeros((self.config['feature_dims'], self.t_out_test), dtype=torch.float32, device=self.device)
        lat_weight_acc = torch.zeros((self.config['feature_dims'], self.t_out_test), dtype=torch.float32, device=self.device)

        # for data in dataset.create_dict_iterator():
        for i, data in enumerate(data_loader):
            # inputs = data['inputs']
            # labels = data['labels']

            # inputs, labels, climates  = map(lambda x: x.to(self.device, dtype = torch.float), data)
            inputs, labels  = map(lambda x: x.to(self.device, dtype = torch.float), data)

            # self.climates = climates
            # inputs, labels = data

            # inputs = inputs.to(self.device, dtype = torch.float)
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
        # lat_weight_rmse = np.sqrt(lat_weight_rmse / (data_length * self.w_size * self.h_size))
        lat_weight_acc = (lat_weight_acc / data_length).cpu().numpy()

        lat_weight_rmse = (lat_weight_rmse / data_length).cpu().numpy()
        lat_weight_rmse = lat_weight_rmse.transpose(1, 0)
        denormalized_lat_weight_rmse = lat_weight_rmse * self.total_std
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
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        total_rmse, total_acc = self.forecast(inputs, labels)

        return total_rmse, total_acc

    def get_position(self):
        num_feature = len(higher_features)
        num_level = len(pressure_level)
        num_surface = len(surface_features)

        if num_surface > 0:
            assert self.config['num_exports'] == num_feature+1, 'num_expert should be equal to num_feature + 1'
        else:
            assert self.config['num_exports'] == num_feature, 'num_expert should be equal to num_feature'

        inputs = torch.zeros([self.config['num_exports'], num_level * num_feature + num_surface])
        for i in range(num_feature):
            inputs[i, i*num_level:(i+1)*num_level] = torch.ones(num_level)
        if num_surface > 0:
            inputs[-1, -num_surface:] = torch.ones(num_surface)
        
        return inputs



    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        # self.h_size * np.cos(PI / 180. * self._lat(j)) / s
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _calculate_lat_weighted_error(self, label, prediction):
        """calculate latitude weighted error"""
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1, 1)
        error = np.square(label - prediction) # the index 0 of label shape is batch_size
        # logging.info(f'error shape: {error.shape}, grid_node_weight shape: {grid_node_weight.shape}')
        lat_weight_error = np.sum(error * grid_node_weight, axis=2)
        lat_weight_error = np.sum(lat_weight_error, axis=0)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """calculate latitude weighted acc"""
        # prediction = prediction * self.std_all.reshape((1, 1, -1)) + self.mean_all.reshape((1, 1, -1))
        # label = label * self.std_all.reshape((1, 1, 1, -1)) + self.mean_all.reshape((1, 1, 1, -1))

        # prediction = prediction - self.climates
        # label = label - self.climates
        prediction = prediction - self.climate
        label = label - self.climate

        prediction = prediction * self.std_all + self.mean_all
        label = label * self.std_all + self.mean_all
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        acc_numerator = np.sum(prediction * label * grid_node_weight, axis=2)
        # acc_denominator = np.sqrt(np.sum(prediction ** 2 * grid_node_weight,
        #                                  axis=2) * np.sum(label ** 2 * grid_node_weight, axis=2))
        acc_denominator = np.sqrt(np.sum(grid_node_weight * prediction ** 2,
                                         axis=2) * np.sum(grid_node_weight * label ** 2, axis=2))
        # acc_numerator = np.sum(acc_numerator, axis=0)
        # acc_denominator = np.sum(acc_denominator, axis=0)

        try:
            # acc = acc_numerator / acc_denominator
            acc = np.divide(acc_numerator, acc_denominator)
            acc = np.sum(acc, axis=0)
        except ZeroDivisionError as e:
            print(repr(e))
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/vamoe.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    # params['interp'] = args.interp
    params['global_batch_size'] = params.batch_size
    # params['global_batch_size'] = 32

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    # vis = args.vis

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    params['surface_features'] = surface_features
    params['higher_features'] =  higher_features
    params['pressure_level'] = pressure_level

    params['old_surface_feature'] = [] 
    # self.old_surface_feature = ['msl', 't2m', 'u10', 'v10']  
    params['old_higher_features'] = ['z', 'q', 'u', 'v', 't']
    params['old_pressure_level'] = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # model = VAMoE(img_size = (params.h_size, params.w_size),
    #             in_chans = params.feature_dims,
    #             out_chans = params.feature_dims,
    #             embed_dim = 768,
    #             num_layers = 16).to(device) 
    model = VAMoE(params, mlp_ratio=params['mlp_ratio']).to(device)

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    test_data_loader, test_dataset = get_data_loader_npy(params, False, run_mode='test')
    logging.info(f"Test dataset size: {len(test_dataset)}")

    start_time = time.time()
    inference_module = InferenceModule(model, params, test_dataset, run_mode='test', device=device)
    inference_module.eval(test_data_loader)
    
    logging.info(f"End-to-End total time: {time.time() - start_time} s")
