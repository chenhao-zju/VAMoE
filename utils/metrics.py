import torch
import pdb
import numpy as np

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor, data_mask: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))

    if data_mask is not None:
        result = torch.sqrt(torch.sum(weight * ((pred - target)*data_mask)**2, dim=(-1,-2))/(data_mask.sum()+1e-10))
        return result
    else:
        result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
        return result

def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
        result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        return result
    
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))
        northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # northern_result = torch.sqrt(torch.mean(northern_weight * (pred[:, :, northern_index:] - clim_time_mean_daily[:, :, northern_index:])**2., dim=(-1,-2)))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))
        southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # southern_result = torch.sqrt(torch.mean(southern_weight * (pred[:, :, :souther_index] - clim_time_mean_daily[:, :, :souther_index])**2., dim=(-1,-2)))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))
        tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
        # tropics_result = torch.sqrt(torch.mean(tropics_weight * (pred[:, :, souther_index:northern_index] - clim_time_mean_daily[:, :, souther_index:northern_index])**2., dim=(-1,-2)))
        return tropics_result
    else:
        raise NotImplementedError

def type_weighted_bias_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    northern_index = int(110. / 180. * num_lat + 0.5)
    souther_index = int(70. / 180. * num_lat + 0.5)


    if metric_type == "all":
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))

        result = torch.mean(weight * pred, dim=(-1,-2))

        return result
    elif metric_type == "northern":
        northern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[northern_index:])
        northern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[northern_index:], souther_index, num_lat, northern_s), (1, 1, -1, 1))


        northern_result = torch.mean(northern_weight * pred[:, :, northern_index:], dim=(-1,-2))
        return northern_result
    elif metric_type == "southern":
        southern_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[:souther_index])
        southern_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[:souther_index], souther_index, num_lat, southern_s), (1, 1, -1, 1))


        southern_result = torch.mean(southern_weight * pred[:, :, :souther_index], dim=(-1,-2))
        return southern_result
        
    elif metric_type == "tropics":
        tropics_s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat))[souther_index:northern_index])
        tropics_weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t[souther_index:northern_index], (northern_index - souther_index), num_lat, tropics_s), (1, 1, -1, 1))


        tropics_result = torch.mean(tropics_weight * pred[:, :, souther_index:northern_index], dim=(-1,-2))
        return tropics_result
    else:
        raise NotImplementedError


def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor, data_mask: torch.Tensor) -> torch.Tensor:
    # import pdb
    # pdb.set_trace()
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    
    if data_mask is not None:
        numerator  = torch.sum(weight * pred*data_mask* target, dim=(-1,-2))
        denominator = torch.sqrt(torch.sum(weight * pred*data_mask* pred, dim=(-1,-2)) * torch.sum(weight * target* data_mask* target, dim=(-1,-2)))
        return numerator/denominator
    else:
        # import pdb
        # pdb.set_trace()
        result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred **2, dim=(-1,-2)) * torch.sum(weight * target ** 2, dim=(-1,-2)))
        return result

def get_hit_miss_counts(pred: torch.Tensor, target:torch.Tensor, mask:torch.Tensor, thresholds=0.5):
        """This function calculates the overall hits and misses for the prediction, which could be used
        to get the skill scores and threat scores:


        This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
        and all inputs should be between 0~1

        Parameters
        ----------
        prediction : torch.Tensor
            Shape: (batch_size, 1, height, width)
        truth : torch.Tensor
            Shape: (batch_size, 1, height, width)
        mask : torch.Tensor or None
            Shape: (batch_size, 1, height, width)
            0 --> not use
            1 --> use
        thresholds : float, list or tuple

        Returns
        -------
        hits : torch.Tensor
            (len(thresholds)) or (batch_size, len(thresholds))
            TP
        misses : torch.Tensor
            (len(thresholds)) or (batch_size, len(thresholds))
            FN
        false_alarms : torch.Tensor
            (len(thresholds)) or (batch_size, len(thresholds))z
            FP
        correct_negatives : torch.Tensor
            (len(thresholds)) or (batch_size, len(thresholds))
            TN
        """

        bpred = (pred >= thresholds)
        btruth = (target >= thresholds)
        bpred_n = torch.logical_not(bpred)
        btruth_n = torch.logical_not(btruth)

        if mask is None:
            hits = torch.logical_and(bpred, btruth).sum(axis=(2,3))
            misses = torch.logical_and(bpred_n, btruth).sum(axis=(2,3))
            false_alarms = torch.logical_and(bpred, btruth_n).sum(axis=(2,3))
            correct_negatives = torch.logical_and(bpred_n, btruth_n).sum(axis=(2,3))
        else:

            hits = torch.logical_and(torch.logical_and(bpred, btruth), mask).sum(axis=(2,3))
            misses = torch.logical_and(torch.logical_and(bpred_n, btruth), mask).sum(axis=(2,3))
            false_alarms = torch.logical_and(torch.logical_and(bpred, btruth_n), mask).sum(axis=(2,3))
            correct_negatives = torch.logical_and(torch.logical_and(bpred_n, btruth_n), mask).sum(axis=(2,3))

        return hits, misses, false_alarms, correct_negatives
    
class Metrics(object):
    """
    Define metrics for evaluation, metrics include:
        - MSE, masked MSE;
        - RMSE, masked RMSE;
        - REL, masked REL;
        - MAE, masked MAE;
        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, **kwargs):
        """
        Initialization.
        Parameters
        ----------
        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon


    def CSI_025(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=2.5)
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)

        return CSI_ACC
    def CSI_0625(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=6.25, )
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)
        return CSI_ACC
    
    def CSI_125(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None,**args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=12.5)
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)
        return CSI_ACC
    
    def CSI_250(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=25.0)
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)
        return CSI_ACC

    def CSI_625(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=62.5)
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)

        return CSI_ACC
    def CSI_700(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        CSI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The CSI metric.
        """

        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, thresholds=70)
        CSI_ACC = TP * 1.0 / (TP + FN + FP + self.epsilon)

        return CSI_ACC

    def SEDI(self, pred, gt, data_mask, clim_time_mean_daily, data_norm, threshold, **args):
        """
        SEDI metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The SEDI metric.
        """
        TP, FN, FP, TN = get_hit_miss_counts(pred, gt, data_mask, data_norm, threshold)

        F = FP * 1.0 / (FP + TN)
        H = TP * 1.0 / (TP + FN)

        SEDI_ACC = (torch.log(F) - torch.log(H) - torch.log(1-F) + torch.log(1-H)) / (torch.log(F) + torch.log(H) + torch.log(1-F) + torch.log(1-H) + self.epsilon)
        return SEDI_ACC


    def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        MSE metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth
        Returns
        -------
        The MSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim=[1,2,3])
        return sample_mse

    def Channel_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        channel_mse = torch.mean((pred - gt) ** 2, dim=[2,3])
        return channel_mse

    def Position_MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        position_mse = torch.mean((pred - gt) ** 2, dim=[0, 1]).reshape(-1)
        return position_mse
    
    def Activity(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        Activity metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth
        data_std: when gt and pred is de-normlized, it keeps as None.
        Returns
        -------
        """
        if data_std is None:
            result = type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all")
        else:
            result = type_weighted_activity_torch_channels(pred - clim_time_mean_daily, metric_type="all")* data_std
        
        return result

    def Bias(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        if data_std is None:
            result = type_weighted_bias_torch_channels(pred-gt, metric_type='all')
        else:
            result = type_weighted_bias_torch_channels(pred-gt, metric_type='all')*data_std
        return  result
    
    def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        RMSE metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The RMSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim = [2,3])
        if data_std is None:
            return torch.sqrt(sample_mse)
        else:
            return torch.sqrt(sample_mse)*data_std
    
    def NRMSE(self, pred, gt,data_mask, clim_time_mean_daily, data_std=None, **args):
        border = 30
        assert pred.size(-1)>2*border and pred.size(-2)>2*border
        
        sample_mse = torch.mean((pred[:,:,border:-border, border:-border] 
                                 - gt[:,:,border:-border, border:-border]) ** 2, 
                                dim=[2,3])
        return torch.sqrt(sample_mse) * data_std
    
    
        
    def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
        """
        MAE metric.
        Parameters
        ----------
        pred: tensor, required, the predicted
        gt: tensor, required, the ground-truth
        Returns
        -------
        
        The MAE metric.
        """
        if data_std is not None:
            return torch.mean(torch.abs(pred - gt), dim=(-1,-2)) * data_std
        else:
            
            return torch.mean(torch.abs(pred - gt), dim=(-1,-2))
    
    def POEP(self, pred, gt, data_mask, clim_time_mean_daily, data_std=None, **args):
            
        threshold = args.get('threshold_per_channel')
        assert   threshold.shape[1] == pred.shape[1]
        absolute_error = pred-gt
        fraction_list=[]
        for i in range(threshold.shape[-1]): # the number of thresholds
            threshold_ = threshold[:,:,i][:,:,None,None]
            if data_std is not None:
                threshold_ =  threshold_/data_std[None, :, None, None]

            has_negative = torch.any(threshold_ < 0)
            if has_negative:
                binary_mask = torch.logical_and( absolute_error<threshold_, absolute_error < 0).sum(dim=(-1,-2))
            else:
                binary_mask = torch.logical_and(absolute_error > threshold_, absolute_error>0).sum(dim=(-1,-2))
            mask_portion =binary_mask/(pred.shape[-2]*pred.shape[-1])
            fraction_list.append(mask_portion[None])
            
            
            # import pdb
            # pdb.set_trace()  
        return torch.cat(fraction_list, axis=0)

    def energy(self, pred):
        data_mean = self.data_mean.to(pred.device)
        data_std = self.data_std.to(pred.device)
        pred = pred * data_std.view(1, pred.shape[1], 1, 1) + data_mean.view(1, pred.shape[1], 1, 1)

        num_lat = pred.shape[2]
        lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
        s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
        weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    
        pred_u = pred[:,43:56]*weight
        pred_v = pred[:,56:69]*weight
        pred_t = pred[:,17:30]*weight

        c_p = 1004

        KED = 0.5*(pred_u**2+pred_v**2)
        TED = c_p/(2*270)*(pred_t**2)

        return (KED + TED).mean().item()

    def spectrum(self, pred, gt, data_mask=None, clim_time_mean_daily=None, data_std=None, **args):
        _, _, num_latitude, num_longitude = pred.shape
        """
        Computes zonal power at wavenumber and frequency.
        """ 
        
        def simple_power(f_x):
            # print(f_x)
            f_k = torch.fft.rfft(f_x, dim=-1)/ num_longitude
            # print(f_k)
            # freq > 0 should be counted twice in power since it accounts for both
            # positive and negative complex values.
            one_and_many_twos = torch.concatenate((torch.tensor([1]), torch.tensor([2] * (f_k.shape[-1] - 1)))).to(f_x)
            result = torch.real(f_k * torch.conj(f_k)) * one_and_many_twos
            return result

        def _circumference(latitude):
            """Earth's circumference as a function of latitude."""
            circum_at_equator = 2 * np.pi * 1000 * (6357 + 6378) / 2
            result = torch.cos(latitude * torch.pi / 180) * circum_at_equator
            return result

        data_mean = self.data_mean.to(pred.device)
        data_std = self.data_std.to(pred.device)
        pred_real = pred * data_std.view(1, pred.shape[1], 1, 1) + data_mean.view(1, pred.shape[1], 1, 1)
        latitude = torch.linspace(-90, 90, num_latitude).to(pred_real)
        spectrum = simple_power(pred_real) # torch.Size([2, 69, 128, 12
        return  spectrum
    
    def WRMSE(self, pred, gt, data_mask=None, clim_time_mean_daily=None, data_std=None, **args):
        """
        WRMSE metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The WRMSE metric.
        """
        if data_std is not None:
            return weighted_rmse_torch(pred, gt, data_mask) * data_std
        else:
            return weighted_rmse_torch(pred, gt, data_mask)

    def WACC(self, pred, gt, data_mask=None, clim_time_mean_daily=None, data_std=None):
        """
        WACC metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The WACC metric.
        """

        return weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily,  data_mask)

    def Var_RMSE(self, pred, gt, channel_to_vname, data_mask, clim_time_mean_daily, data_std=None):
        """
        WRMSE metric.
        Parameters
        ----------
        pred: tensor, required, the predicted;
        gt: tensor, required, the ground-truth;
        Returns
        -------
        The WRMSE metric.
        """
        
        return weighted_rmse_torch(pred, gt) * data_std



## Not used in this version
class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
        """
        Initialization.
        Parameters
        ----------
        metrics_list: list of str, required, the metrics name list used in the metric calcuation.
        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon = epsilon)
        self.metrics_list = []
        for metric in metrics_list:
            try:

                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                raise NotImplementedError('Invalid metric type. Please set the right\
                                          metric name in config file')

    def evaluate_batch(self, data_dict):
        """
        Evaluate a batch of the samples.
        Parameters
        ----------
        data_dict: pred and gt
        Returns
        -------
        The metrics dict.
        """
        pred = data_dict['pred']            # (B, C, H, W)
        gt = data_dict['gt']
        channels_to_vname = data_dict['channels_to_vname']
        data_mask = None
        clim_time_mean_daily = None
        data_std = None
        if "clim_mean" in data_dict:
            clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
            data_std = data_dict["std"]

        losses = {}

        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std)

            import pdb
            pdb.set_trace()

            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name+ '_'+ channels_to_vname[i]] = loss[i].item()
            else:
                losses[metric_name] = loss

        return losses