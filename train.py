import os
import time
import numpy as np
import argparse
# import h5py
import torch
import cProfile
import re
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
# from utils.data_loader_multifiles import get_data_loader
from utils.data_loader_npyfiles import get_data_loader_npy, surface_features, higher_features, pressure_level
from networks import VAMoE
from utils.img_utils import vis_precip
import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch
from apex import optimizers
from utils.darcy_loss import LpLoss
from networks.l2_loss import L2_LOSS
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
DECORRELATION_TIME = 36 # 9 days
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from einops import rearrange

from test import InferenceModule


class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if params.log_to_wandb:
            # wandb.init(config=params, name=params.name, group=params.group, project=params.project, entity=params.entity)
            wandb.init(config=params, name=params.name, group=params.group, project=params.project)

        logging.info('rank %d, begin data loader init'%world_rank)
        # self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
        # self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader_npy(params, dist.is_initialized(), run_mode='train')
        self.valid_data_loader, self.valid_dataset = get_data_loader_npy(params, dist.is_initialized(), run_mode='valid')

        self.loss_type = params['loss']
        self.loss_weight = params['loss_weight']

        logging.info(f'****** using {self.loss_type} in model training ******')
        if self.loss_type == 'l2':
            learn_log_variance=dict(flag=True, channels=params['feature_dims'], logvar_init=0., requires_grad=True)
            self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)
            self.loss_recons = L2_LOSS(learn_log_variance=learn_log_variance).to(self.device)

        elif self.loss_type == 'lploss':
            self.loss_gen = LpLoss()
            self.loss_recons = LpLoss()

        logging.info('rank %d, data loader initialized'%world_rank)

        # params.crop_size_x = self.valid_dataset.crop_size_x
        # params.crop_size_y = self.valid_dataset.crop_size_y
        # params.img_shape_x = self.valid_dataset.img_shape_x
        # params.img_shape_y = self.valid_dataset.img_shape_y

        # precip models
        # self.precip = True if "precip" in params else False
        self.precip = False
        self.use_moe = params['use_moe']
        self.use_cl = params['use_cl']
        self.mlp_ratio = params['mlp_ratio']

        self.surface_features = params['surface_features'] = surface_features
        self.higher_features = params['higher_features'] =  higher_features
        self.pressure_level =params['pressure_level'] = pressure_level

        self.old_surface_feature = params['old_surface_feature'] = [] 
        # self.old_surface_feature = ['msl', 't2m', 'u10', 'v10']  
        self.old_higher_features = params['old_higher_features'] = ['z', 'q', 'u', 'v']
        self.old_pressure_level = params['old_pressure_level'] = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

        # self.model = VAMoE(img_size = (params.h_size, params.w_size),
        #                   in_chans = params.feature_dims,
        #                   out_chans = params.feature_dims,
        #                   embed_dim = 768,
        #                   num_layers = 16).to(self.device) 
        self.model = VAMoE(params, mlp_ratio=self.mlp_ratio).to(self.device)

        if self.params.enable_nhwc:
            # NHWC: Convert model to channels_last memory format
            self.model = self.model.to(memory_format=torch.channels_last)

        if params.log_to_wandb:
            wandb.watch(self.model)

        # fix optimizer to adamw  08.21
        # logging lr
        # guarantee 5e-4 to 1e-6
        # lr decay strategy: cos decay 
        if params.optimizer_type == 'FusedAdam':
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)
        elif params.optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s"%params.checkpoint_path)
            with torch.no_grad():
                self.restore_checkpoint(params.checkpoint_path)

        if self.use_cl and self.startEpoch == 0:
            logging.info("Loading continuous learning checkpoint %s"%params.cl_ckpt_path)
            with torch.no_grad():
                self.load_checkpoint_cl(params.cl_ckpt_path)

        if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
            self.posembed = self.get_position()

        if dist.is_initialized():
            # self.model = DistributedDataParallel(self.model,
            #                                   device_ids=[params.local_rank],
            #                                   output_device=[params.local_rank],find_unused_parameters=True)
            self.model = DistributedDataParallel(self.model,
                                              device_ids=[params.local_rank],
                                              output_device=[params.local_rank],find_unused_parameters=False)

        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch-1, eta_min=params.min_lr)
        else:
            self.scheduler = None

        '''if params.log_to_screen:
          logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

        self.inference = InferenceModule(self.model, self.params, self.valid_dataset, run_mode='valid')

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def loss_obj(self, pred1, pred2, target1, target2):
        loss1 = self.loss_gen(pred1, target1)
        loss2 = self.loss_recons(pred2, target2)
        return (1-self.loss_weight)*loss1 + self.loss_weight*loss2

    def get_position(self):
        num_feature = len(higher_features)
        num_level = len(pressure_level)
        num_surface = len(surface_features)

        if num_surface > 0:
            assert self.params['num_exports'] == num_feature+1, 'num_expert should be equal to num_feature + 1'
        else:
            assert self.params['num_exports'] == num_feature, 'num_expert should be equal to num_feature'

        inputs = torch.zeros([self.params['num_exports'], num_level * num_feature + num_surface])
        for i in range(num_feature):
            inputs[i, i*num_level:(i+1)*num_level] = torch.ones(num_level)
        if num_surface > 0:
            inputs[-1, -num_surface:] = torch.ones(num_surface)
        
        return inputs

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
                # self.valid_sampler.set_epoch(epoch)
            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()
            if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
                valid_weighted_rmse = self.validate_final()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})


            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if self.epoch % 5 == 0:
                        mid_checkpoint_path = self.params.mid_checkpoint_path + f'ckpt{str(self.epoch)}.tar'
                        os.system(f'cp {self.params.checkpoint_path} {mid_checkpoint_path}')

                        if len(os.listdir(self.params.mid_checkpoint_path)) >= 4:
                            name = mid_checkpoint_path.split('/')[-1]
                            num = int(name.split('.')[0][4:]) - 5
                            rm_checkpoint_path = self.params.mid_checkpoint_path + f'ckpt{str(num)}.tar'
                            os.system(f'rm -r {rm_checkpoint_path}')

                    if valid_logs['valid_loss'] <= best_valid_loss:
                        #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']


            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                #logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info('Train loss: {}. Valid loss: {}. Learning Rate: {}.'.format(train_logs['loss'], valid_logs['valid_loss'], current_lr))
                # logging.info(f"Test results of RMSE: z500: {valid_logs['z500']}, t2m: {valid_logs['t2m']}, t850: {valid_logs['t850']}, u10: {valid_logs['u10']}")
                try:
                    logging.info(f"Test results of RMSE: z500: {valid_logs['z500']}, q500: {valid_logs['q500']}, u500: {valid_logs['u500']}")
                except:
                    logging.info(f"Test results of RMSE: z500: {valid_logs['z500']}, q500: {valid_logs['q500']}")


        #        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
        #          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))



    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()
        
        for i, data in enumerate(self.train_data_loader):
            self.iters += 1
            # adjust_LR(optimizer, params, iters)
            data_start = time.time()
            inp, target = map(lambda x: x.to(self.device, dtype = torch.float), data)      
            if self.params.orography and self.params.two_step_training:
                orog = inp[:,-2:-1] 

            data_time += time.time() - data_start

            tr_start = time.time()

            t_out_train = self.params['t_out_train']
            for j in range(t_out_train):
                # logging.info(f"multi outputs training: {t_out_train}, number of loop: {j}, target: {target.shape}, {inp[0,0,10, 10:20]}")

                if t_out_train == 1:
                    tar = target.clone()
                else:
                    if j>0:   inp = gen.detach()
                    tar = target[:, j]

                # logging.info(f"changed input: {inp[0,0,10, 10:20]}")

                self.model.zero_grad()
                with amp.autocast(self.params.enable_amp):
                    if self.loss_type == 'trainl2':
                        if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                            self.posembed = self.posembed.to(self.device, dtype = torch.float)
                            gen, recons, loss = self.model(inp, target=tar, posembed=self.posembed, run_mode='train')
                        else:
                            gen, recons, loss = self.model(inp, target=tar, run_mode='train')
                    else:                    
                        if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                            self.posembed = self.posembed.to(self.device, dtype = torch.float)
                            results = self.model(inp, posembed=self.posembed, run_mode='train')
                        else:
                            results = self.model(inp, run_mode='train')    # .to(self.device, dtype = torch.float)
                        if self.use_moe=='moe':
                            gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), results[:-1])     
                            loss = self.loss_obj(gen, recons, tar, inp) + results[-1]
                        else:
                            gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), results)     
                            loss = self.loss_obj(gen, recons, tar, inp)

                # if j == t_out_train - 1:
                if self.params.enable_amp:
                    self.gscaler.scale(loss).backward()
                    self.gscaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.params.enable_amp:
                    self.gscaler.update()

                if self.params['use_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

                tr_time += time.time() - tr_start
        
        try:
            logs = {'loss': loss, 'loss_step_one': loss_step_one, 'loss_step_two': loss_step_two}
        except:
            logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key]/dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 20 #do validation on first 20 images, just for LR scheduler

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

        valid_start = time.time()

        # self.inference.model = self.model

        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                if (not self.precip) and i>=n_valid_batches:
                    break    
                # inp, tar, climate  = map(lambda x: x.to(self.device, dtype = torch.float), data)
                inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)
                if self.params.orography and self.params.two_step_training:
                    orog = inp[:,-2:-1]
                
                if self.loss_type == 'trainl2':
                    if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        self.posembed = self.posembed.to(self.device, dtype = torch.float)
                        gen, recons, valid_loss = self.model(inp, target=tar, posembed=self.posembed, run_mode='val')
                    else:
                        gen, recons, valid_loss = self.model(inp, target=tar, run_mode='val')
                    gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), [gen, recons])    

                else:                    
                    if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        self.posembed = self.posembed.to(self.device, dtype = torch.float)
                        results = self.model(inp, posembed=self.posembed, run_mode='val')
                    else:
                        results = self.model(inp, run_mode='val')    # .to(self.device, dtype = torch.float)

                    if self.use_moe=='moe':
                        gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), results[:-1])     
                        valid_loss += self.loss_obj(gen, recons, tar, inp) + results[-1]
                    else:
                        gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), results)     
                        valid_loss += self.loss_obj(gen, recons, tar, inp) 
                valid_l1 += nn.functional.l1_loss(gen, tar)

                valid_steps += 1.
                # save fields for vis before log norm 
                if (i == sample_idx) and (self.precip and self.params.log_to_wandb):
                    fields = [gen[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]

                if self.precip:
                    gen = unlog_tp_torch(gen, self.params.precip_eps)
                    tar = unlog_tp_torch(tar, self.params.precip_eps)

                valid_weighted_rmse += weighted_rmse_torch(gen, tar)

                if not self.precip:
                    try:
                        os.mkdir(params['experiment_dir'] + "/" + str(i))
                    except:
                        pass
                    #save first channel of image
                    if self.params.two_step_training:
                        save_image(torch.cat((gen_step_one[0,0], torch.zeros((self.valid_dataset.h_size, 4)).to(self.device, dtype = torch.float), tar[0,0]), axis = 1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")
                    else:
                        save_image(torch.cat((gen[0,0], torch.zeros((self.valid_dataset.h_size, 4)).to(self.device, dtype = torch.float), tar[0,0]), axis = 1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")

              
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        # if not self.precip:
        #   valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = self.inference.total_std * valid_weighted_rmse.detach().cpu().numpy()


        valid_time = time.time() - valid_start
        # valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)
        # valid_weighted_rmse = torch.mean(valid_weighted_rmse, axis = 0)
        if self.precip:
            logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_tp': valid_weighted_rmse_cpu[0]}
        else:
            num_surface_variables = len(self.surface_features)

            # logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'z500': valid_weighted_rmse_cpu[5], 't2m': valid_weighted_rmse_cpu[-3], 't850': valid_weighted_rmse_cpu[54], 'u10': valid_weighted_rmse_cpu[-2]}
            try:
                logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'z500': valid_weighted_rmse_cpu[5], 'q500': valid_weighted_rmse_cpu[18], 'u500': valid_weighted_rmse_cpu[31] }
            except:
                logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'z500': valid_weighted_rmse_cpu[5], 'q500': valid_weighted_rmse_cpu[18]}

            for i, name in enumerate(self.surface_features):
                logs[name] = valid_weighted_rmse_cpu[i-num_surface_variables]

        if self.params.log_to_wandb:
            if self.precip:
                fig = vis_precip(fields)
                logs['vis'] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    def validate_final(self):
        self.model.eval()
        n_valid_batches = int(self.valid_dataset.n_patches_total/self.valid_dataset.n_patches) #validate on whole dataset
        valid_weighted_rmse = torch.zeros(n_valid_batches, self.params.N_out_channels)

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)

        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                if i>100:
                    break
                inp, tar, _ = map(lambda x: x.to(self.device, dtype = torch.float), data)
                if self.params.orography and self.params.two_step_training:
                    orog = inp[:,-2:-1]
                if 'residual_field' in self.params.target:
                    tar -= inp[:, 0:tar.size()[1]]
            
                if self.loss_type == 'trainl2':
                    if self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        self.posembed = self.posembed.to(self.device, dtype = torch.float)
                        gen, recons, loss = self.model(inp, target=tar, posembed=self.posembed, run_mode='val')
                    else:
                        gen, recons, loss = self.model(inp, target=tar, run_mode='val')
                    gen, recons = map(lambda x: x.to(self.device, dtype = torch.float), [gen, recons])    
                    valid_loss[i] += loss

                else:  
                    if self.use_moe=='moe':
                        gen, recons, l = self.model(inp, run_mode='val')
                        valid_loss[i] += self.loss_obj(gen, recons, tar, inp) + l
                    elif self.use_moe == 'densemoe' or self.use_moe == 'channelmoe' or self.use_moe=='channelmoev1' or self.use_moe=='channelmoev3':
                        self.posembed = self.posembed.to(self.device, dtype = torch.float)
                        gen, recons = self.model(inp, self.posembed, run_mode='val')
                        valid_loss[i] += self.loss_obj(gen, recons, tar, inp) 
                    else:
                        gen, recons = self.model(inp, run_mode='val')
                        valid_loss[i] += self.loss_obj(gen, recons, tar, inp) 
                valid_l1[i] += nn.functional.l1_loss(gen, tar)

                for c in range(self.params.N_out_channels):
                    if 'residual_field' in self.params.target:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch((gen[0,c] + inp[0,c]), (tar[0,c]+inp[0,c]), self.device)
                    else:
                        valid_weighted_rmse[i, c] = weighted_rmse_torch(gen[0,c], tar[0,c], self.device)
                
            #un-normalize
            # valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse[0:100], axis = 0).to(self.device)
            valid_weighted_rmse = torch.mean(valid_weighted_rmse[0:100], axis = 0).to(self.device)

        return valid_weighted_rmse


    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        if params['checkpoint'] == '':
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']

        if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_checkpoint_cl(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        static_state = checkpoint['model_state']

        # self.old_surface_feature = [] 
        # # self.old_surface_feature = ['msl', 't2m', 'u10', 'v10']  
        # self.old_higher_features = ['z', 'q', 'u', 'v', 't']
        # self.old_pressure_level = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

        self.new_surface_feature = surface_features
        self.new_higher_features = higher_features
        self.new_pressure_level = pressure_level

        self.proj_param = [self.new_higher_features.index(i)*len(self.new_pressure_level)+self.new_pressure_level.index(j) for i in self.old_higher_features for j in self.old_pressure_level] + [self.new_surface_feature.index(k)-len(self.new_surface_feature) for k in self.old_surface_feature]

        model_state = self.model.state_dict()
        # logging.info(self.model_state.keys())
        # self.new_proj = self.model_state['patch_embed.proj.weight']
        self.proj = static_state['module.patch_embed.proj.weight']
        self.new_proj = model_state['patch_embed.proj.weight']

        self.head = static_state['module.head.weight']
        self.head = rearrange(self.head, "(n c1) c2 -> n c1 c2", c1=self.proj.shape[1])
        self.new_head = model_state['head.weight']
        self.new_head = rearrange(self.new_head, "(n c1) c2 -> n c1 c2", c1=self.new_proj.shape[1])

        self.posembedder = static_state['module.posembedder.mlp.0.weight']
        self.new_posembedder = model_state['posembedder.mlp.0.weight']
        
        self.loss_weight1 = static_state['module.loss_recons.logvar']
        self.new_loss_wight1 = model_state['loss_recons.logvar']

        self.loss_weight2 = static_state['module.loss_gen.logvar']
        self.new_loss_wight2 = model_state['loss_gen.logvar']

        # logging.info(f'mask: , {self.proj_param}')
        # logging.info(f'ckpt shape: , {self.proj.shape}, {self.new_proj.shape}, {len(self.proj_param)}')
        # logging.info(f'ckpt shape: , {self.head.shape}, {self.new_head.shape}')
        # logging.info(f'new_proj features: , {self.new_proj[0,:,0,0]}')
        # logging.info(f'proj features: , {self.proj[0,:,0,0]}')
        # logging.info(f'new_head features: , {self.new_head[0,:,0]}')
        # logging.info(f'head features: , {self.head[0,:,0]}')

        for i, index in enumerate(self.proj_param):
            self.new_proj[:, index] = self.proj[:, i]
            self.new_head[:, index] = self.head[:, i]
            self.new_posembedder[:, index] = self.posembedder[:, i]
            self.new_loss_wight1[:, index] = self.loss_weight1[:, i]
            self.new_loss_wight2[:, index] = self.loss_weight2[:, i]

        # logging.info(f'new_proj features: , {self.new_proj[0,:,0,0]}')
        # logging.info(f'new_head features: , {self.new_head[0,:,0]}')
        self.new_head = rearrange(self.new_head, "n c1 c2 -> (n c1) c2")

        new_state_dict = model_state
        key_lists = list(new_state_dict.keys())
        for key, val in static_state.items():
            name = key[7:]
            if name in key_lists:
                if name == 'patch_embed.proj.weight':
                    new_state_dict[name] = self.new_proj
                elif name == 'head.weight':
                    new_state_dict[name] = self.new_head
                elif name == 'posembedder.mlp.0.weight':
                    new_state_dict[name] = self.new_posembedder
                elif name == 'loss_recons.logvar':
                    new_state_dict[name] = self.new_loss_wight1
                elif name == 'loss_gen.logvar':
                    new_state_dict[name] = self.new_loss_wight2
                else:
                    new_state_dict[name] = val 
        self.model.load_state_dict(new_state_dict)

        for name, param in self.model.named_parameters():
            # if 'filter' in name or 'layer_norm' in name:
            if 'blocks' in name:
                if 'norm1' in name or 'attn' in name or 'norm2' in name:
                    param.requires_grad = False
                # if 'moe.experts.0' in name or 'moe.experts.1' in name or 'moe.experts.2' in name or 'moe.experts.3' in name or 'moe.experts.4' in name:
                if 'moe.experts.0' in name or 'moe.experts.1' in name or 'moe.experts.2' in name or 'moe.experts.3' in name:
                    param.requires_grad = False



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/vamoe.yaml', type=str)
  parser.add_argument("--checkpoint", default='', type=str)
  parser.add_argument("--exp_dir", default='./logs/test', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params['epsilon_factor'] = args.epsilon_factor

  params['world_size'] = 1
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])
  logging.info(f"world_size:, {params['world_size']}")

  wandb.require("core")
  world_rank = 0
  local_rank = 0
  if params['world_size'] > 1:
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            )
    local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = local_rank
    world_rank = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  
  logging.info(f"local_rank:, {local_rank}")

  torch.cuda.set_device(local_rank)
  torch.backends.cudnn.benchmark = True

  # Set up directory
  # expDir = os.path.join(args.exp_dir, args.config, str(args.run_num))
  expDir = args.exp_dir + args.config + '/' + str(args.run_num) + '/'
  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
    #   os.makedirs(os.path.join(expDir, 'training_checkpoints/'))
      os.makedirs(expDir + 'training_checkpoints/')

  params['experiment_dir'] = os.path.abspath(expDir)
#   params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
#   params['mid_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/')
#   params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
  params['checkpoint_path'] = expDir + 'training_checkpoints/ckpt.tar'
  params['mid_checkpoint_path'] = expDir + 'training_checkpoints/'
  params['best_checkpoint_path'] = expDir + 'training_checkpoints/best_ckpt.tar'

  params['checkpoint'] = args.checkpoint

  if args.checkpoint != '':
    params['checkpoint_path'] = args.checkpoint

  # Do not comment this line out please:
  args.resuming = True if os.path.isfile(params.checkpoint_path) else False

  params['resuming'] = args.resuming
  params['local_rank'] = local_rank
  params['enable_amp'] = args.enable_amp

  # this will be the wandb name
#  params['name'] = args.config + '_' + str(args.run_num)
#  params['group'] = "era5_wind" + args.config
  params['name'] = args.config + '_' + str(args.run_num)
  params['project'] = "VAMoE"
  # params['entity'] = "flowgan"
  if world_rank==0:
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    logging_utils.log_versions()
    params.log()

  params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
  params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

  # params['in_channels'] = np.array(params['in_channels'])
  # params['out_channels'] = np.array(params['out_channels'])
  # if params.orography:
  #   params['N_in_channels'] = len(params['in_channels']) +1
  # else:
  #   params['N_in_channels'] = len(params['in_channels'])
  # params['N_out_channels'] = len(params['out_channels'])

  params['in_channels'] = np.array(list(range(params['in_channels'])))
  params['out_channels'] = np.array(list(range(params['out_channels'])))
  if params.orography:
    params['N_in_channels'] = len(params['in_channels']) +1
  else:
    params['N_in_channels'] = len(params['in_channels'])
  params['N_out_channels'] = len(params['out_channels'])

  if world_rank == 0:
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
      hparams[str(key)] = str(value)
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
      yaml.dump(hparams,  hpfile )

  trainer = Trainer(params, world_rank)
  trainer.train()
  logging.info('DONE ---- rank %d'%world_rank)
