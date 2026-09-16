""" 
Created on Monday Jan 26 2026
@author: Azad Md Abulkalam
@location: ISB, NTNU

To load/train/finetune/test EchoTracker/PIPs++/SpeckNet/CpTracker3 in echocardiography.
"""
import torch.nn.functional as F
import torch
from utils.utils_ import getMetricsDict, randomize_points, sequence_loss, requires_grad, fetch_optimizer
from utils import saverloader
from tqdm import tqdm
from utils import evaluate
import os
from torch.utils.tensorboard import SummaryWriter
from models.cotracker.model_utils import get_points_on_a_grid

from models.specknet import SpeckNet
from models.echotracker import EchoTracker
from models.pips2 import Pips as Pips2
from models.cotracker3 import CoTrackerThreeOffline
from models.locotrack import LocoTrack
from models.echotracker2 import EchoTracker2

class ECHOTRACKER2():
    def __init__(
            self, 
            model_size='base', # 'small'
            device_ids=list(range(torch.cuda.device_count())),
            ft_model = False,
            sp_attn = False,
            temp_shift_module = False,
            ct3_trans = False,
            fflow = False,
            local_temp_smoothing = False,
            resnet_temp_module = False,
            patch_size = 7,
            bidirectional_temp_attn = False,
        ) -> None:
        self.ft_model = ft_model
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model_size = model_size
        self.model = EchoTracker2(
            sp_attn=sp_attn,
            itsm_resnet=resnet_temp_module,    
            patch_size=patch_size,
        ).to(self.device)

    def load(self, path=None, eval=True):
        """ load the model with a given trained weigth for training or inferencing

        Args:
            path (str): path to folder containing the trained weight
            eval (bool, optional): False if the model is loaded for training. Defaults to True.

        Returns:
            model: a loaded model with train or eval mode activated
        """
        if path is not None and self.ft_model:
            _ = saverloader.load(path, self.model)
            #_ = saverloader.load(path, self.model, ignore_load='torch_pips_mixer')
        elif path is not None:
            with open(path, "rb") as f:
                #state_dict = torch.load(f, map_location="cpu")
                state_dict = torch.load(f)['state_dict']
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            #self.model.load_state_dict(torch.load(path, map_location="cpu"))
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print("The EchoTracker2 model is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print("The EchoTracker2 model is loaded for training.")
            return self.model.train()   

    def infer(self, video, points, resize=None):
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]
            points (tensor): a list of points across different timesteps [B, N, 3]->(t, x, y). range: [0.0 - 1.0]
            resize (tuple): (H, W)
        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]
        """
        rgbs = video.float().to(self.device)
        #rgbs = (2 * (rgbs / 255.0) - 1.0)
        
        B, S, H, W, C = video.shape
        if resize is not None:
            rgbs_ = rgbs.permute(0, 1, 4, 2, 3).reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W).permute(0, 1, 3, 4, 2)
        

        points[...,1] *= W - 1
        points[...,2] *= H - 1
        query_points = points.to(self.device)
        
        # #if not self.ft_model:
        query_points = query_points[..., [0, 2, 1]] #(t, x, y) -> (t, y, x)

        
        with torch.no_grad():
            torch.cuda.empty_cache()
            # outputs, _ = self.model(rgbs, query_points)
            # print(rgbs.shape, query_points.shape)
            # raise KeyboardInterrupt
            outputs = self.model(rgbs, query_points)

        trajs_e = outputs['tracks']
        
        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        
        points[...,1] /= W - 1
        points[...,2] /= H -1

        return trajs_e
    
    
    def train(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):        
        """ Train the model on the provided dataset.
        Args:
            dataloaders (Dict): {'train':torch train dataloader, 'val':torch train dataloader}
            dataset_size (Dict): {'train': int, 'val':int}
            epoch (int, optional): number of epochs to train. Defaults to 10.

        Returns:
            _type_: _description_
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size['train']
        optimizer, scheduler = fetch_optimizer(
            lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters()
        )
        # metrics = getMetricsDict()
        # metrics['loss'] = 0.0
        best_loss = float('inf')
        best_davg = 0.0
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in tqdm(range(epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                
                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.float().to(self.device)  # [B, T, H, W, C]
                    #rgbs = (2 * (rgbs / 255.0) - 1.0) # normalizing [-1 1]
                    trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)  # [B, N, T, 2]
                    visibs_g = visibs_g.to(self.device)  # [B, T, N]
                    
                    # Normalize trajectories
                    B, S, H, W, C = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1

                    # Select query points from random frames
                    if q_frame == -1 and phase == 'val':
                        fix = int(S * 0.75)
                    elif q_frame == -2:# or phase == 'test':
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix = q_frame

                    #points_0 = trajs_g[:, :, q_frame, :]  # Initial points from q frame
                    q_points = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix).to(self.device)  # [B, N, 3]
                    q_points = q_points[..., [0, 2, 1]] #(t, x, y) -> (t, y, x)
                    # print(q_points[0,0,0])
                    
                    with torch.set_grad_enabled(phase == "train"):
                        # print(rgbs.shape, q_points.shape)
                        # raise KeyboardInterrupt
                        outputs = self.model(video=rgbs, query_points=q_points)                        
                        # print(trajs_e.shape, trajs_g.shape, trajs_e.min(), trajs_e.max(), trajs_g.min(), trajs_g.max())
                        # raise KeyboardInterrupt
                        batch_loss = sequence_loss([outputs['tracks']], trajs_g.permute(0, 2, 1, 3), visibs_g, visibs_g, 0.8).mean() # Mean across the batch
                        # print(loss)
                        # raise KeyboardInterrupt
                    
                        if phase == "train":
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            optimizer.step()
                            scheduler.step()
                        
                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Normalize predictions and ground truth back to [0, 1]
                        trajs_e = outputs['tracks'].permute(0, 2, 1, 3)
                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        # points_0[..., 0] /= W - 1
                        # points_0[..., 1] /= H - 1
                        q_points[..., 1] /= W - 1
                        q_points[..., 2] /= H - 1

                        # Evaluate metrics
                        outputs = evaluate.compute_metrics(
                            # points_0.cpu().numpy(),
                            q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            first=False
                        )
                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)  # Accumulate over batch size
                        
                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, phase, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] <= best_loss:
                        saverloader.save(os.path.join(ckpt_path, phase), optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] >= best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, phase, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()



class LOCOTRACK():
    def __init__(
            self, 
            model_size='base', # 'small'
            device_ids=list(range(torch.cuda.device_count())),
            ft_model = False,
        ) -> None:
        self.ft_model = ft_model
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model_size = model_size
        self.model = LocoTrack(model_size=model_size).to(self.device)

    def load(self, path, eval=True):
        """ load the model with a given trained weigth for training or inferencing

        Args:
            path (str): path to folder containing the trained weight
            eval (bool, optional): False if the model is loaded for training. Defaults to True.

        Returns:
            model: a loaded model with train or eval mode activated
        """
        if path is not None and self.ft_model:
            _ = saverloader.load(path, self.model)
        elif path is not None:
            with open(path, "rb") as f:
                #state_dict = torch.load(f, map_location="cpu")
                state_dict = torch.load(f)['state_dict']
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            #self.model.load_state_dict(torch.load(path, map_location="cpu"))
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print("The LocoTrack model is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print("The LocoTrack model is loaded for training.")
            return self.model.train()   

    def infer(self, video, points, resize=None):
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]
            points (tensor): a list of points across different timesteps [B, N, 3]->(t, x, y). range: [0.0 - 1.0]
            resize (tuple): (H, W)
        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]
        """
        rgbs = video.float().to(self.device)
        # rgbs = (2 * (rgbs / 255.0) - 1.0)
        
        B, S, H, W, C = video.shape
        if resize is not None:
            rgbs_ = rgbs.permute(0, 1, 4, 2, 3).reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W).permute(0, 1, 3, 4, 2)
        

        points[...,1] *= W - 1
        points[...,2] *= H - 1
        query_points = points.to(self.device)
        
        # #if not self.ft_model:
        # query_points = query_points[..., [0, 2, 1]] #(t, x, y) -> (t, y, x)

        
        with torch.no_grad():
            torch.cuda.empty_cache()
            # outputs, _ = self.model(rgbs, query_points)
            # print(rgbs.shape, query_points.shape)
            # raise KeyboardInterrupt
            outputs = self.model(rgbs, query_points)

        trajs_e = outputs['tracks']
        occlusions = outputs['occlusion']
        expected_dist = outputs['expected_dist']
        visibs_e = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5

        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        
        points[...,1] /= W - 1
        points[...,2] /= H -1

        return trajs_e, visibs_e
    
    
    def finetune(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):        
        """ Finetune the model on the provided dataset.
        Args:
            dataloaders (Dict): {'train':torch train dataloader, 'val':torch train dataloader}
            dataset_size (Dict): {'train': int, 'val':int}
            epoch (int, optional): number of epochs to train. Defaults to 10.

        Returns:
            _type_: _description_
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size['train']
        optimizer, scheduler = fetch_optimizer(
            lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters()
        )
        # metrics = getMetricsDict()
        # metrics['loss'] = 0.0
        best_loss = float('inf')
        best_davg = 0.0
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in tqdm(range(epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                
                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.float().to(self.device)  # [B, T, H, W, C]
                    #rgbs = (2 * (rgbs / 255.0) - 1.0) # normalizing [-1 1]
                    trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)  # [B, N, T, 2]
                    visibs_g = visibs_g.to(self.device)  # [B, T, N]
                    
                    # Normalize trajectories
                    B, S, H, W, C = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1

                    # Select query points from random frames
                    if q_frame == -1:
                        fix = int(S * 0.75)
                    elif q_frame == -2:
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix = q_frame

                    #points_0 = trajs_g[:, :, q_frame, :]  # Initial points from q frame
                    q_points = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix).to(self.device)  # [B, N, 3]
                    #q_points = q_points[..., [0, 2, 1]] #(t, x, y) -> (t, y, x)
                    # print(q_points[0,0,0])
                    
                    with torch.set_grad_enabled(phase == "train"):
                        # print(rgbs.shape, q_points.shape)
                        # raise KeyboardInterrupt
                        outputs = self.model(video=rgbs, query_points=q_points)                        
                        # print(trajs_e.shape, trajs_g.shape, trajs_e.min(), trajs_e.max(), trajs_g.min(), trajs_g.max())
                        # raise KeyboardInterrupt
                        batch_loss = sequence_loss([outputs['tracks']], trajs_g.permute(0, 2, 1, 3), visibs_g, visibs_g, 0.8).mean() # Mean across the batch
                        # print(loss)
                        # raise KeyboardInterrupt
                    
                        if phase == "train":
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            optimizer.step()
                            scheduler.step()
                        
                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Normalize predictions and ground truth back to [0, 1]
                        trajs_e = outputs['tracks'].permute(0, 2, 1, 3)
                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        # points_0[..., 0] /= W - 1
                        # points_0[..., 1] /= H - 1
                        q_points[..., 1] /= W - 1
                        q_points[..., 2] /= H - 1

                        # Evaluate metrics
                        outputs = evaluate.compute_metrics(
                            # points_0.cpu().numpy(),
                            q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            first=False
                        )
                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)  # Accumulate over batch size
                        
                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] < best_loss:
                        saverloader.save(ckpt_path, optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] > best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()


class SPECKNET():
    def __init__(
            self,
            device_ids=list(range(torch.cuda.device_count())),
            multi_scale_level = 2,
        ) -> None:

        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model = SpeckNet(multi_scale_level=multi_scale_level,
                        ).to(self.device)

    def load(self, path=None, eval=True):
        """ load the model with a given trained weigth for training or inferencing

        Args:
            path (str): path to folder containing the trained weight
            eval (bool, optional): False if the model is loaded for training. Defaults to True.

        Returns:
            model: a loaded model with train or eval mode activated
        """
        if path is not None:
            _ = saverloader.load(path, self.model)
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print(f"The {self.__class__.__name__} model is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print(f"The {self.__class__.__name__} model is loaded for training.")
            return self.model.train()   

    def infer(self, video, points, resize=None, track_q=False):
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]
            points (tensor): a list of points across different timesteps [B, N, 3]->(t, x, y). range: [0.0 - 1.0]
            resize (tuple): (H, W)
        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]
        """
        #points = points.to(self.device)
        rgbs = video.float().to(self.device)
        B, S, H, W, C = video.shape
        if resize is not None:
            rgbs_ = rgbs.reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W).permute(0, 1, 3, 4, 2)
            

        points[...,1] *= W - 1
        points[...,2] *= H - 1
        #query_points = convert_select_points_to_query_points(frame=0, points=points)# all points are in frame 0
        # from (t, x, y) to (t, y, x)
        query_points = points[:,:,[0, 2, 1]] # (B, N, 3)
        # # preparing the time dimension to be concatenated
        # time_dim = torch.zeros((query_points.shape[0], query_points.shape[1], 1))
        # # prepending a column to be -> (B, N, 3)
        # query_points = torch.concatenate((time_dim, query_points), axis=-1).squeeze()


        #rgbs = (2 * (rgbs / 255.0) - 1.0).to(self.device)
        query_points = query_points.to(self.device)
        # query_points = query_points.unsqueeze(0)

        # print(rgbs.shape, query_points.shape)
        # raise KeyboardInterrupt
        with torch.no_grad():
            outputs = self.model(rgbs, query_points)
            #print(torch.cuda.memory_allocated()/ (1024 ** 2))

        trajs_e = outputs['tracks']
        
        #expected_dist = outputs['expected_dist']
        visibs_e = None#(1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5

        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        
        points[...,1] /= W - 1
        points[...,2] /= H - 1

        if track_q:
            trajs_q = outputs['tracks_q']
            return trajs_e, visibs_e, trajs_q
        else:
            return trajs_e, visibs_e
    
    def train(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):        
        """ Finetune the model on the provided dataset.
        Args:
            dataloaders (Dict): {'train':torch train dataloader, 'val':torch val dataloader}
            dataset_size (Dict): {'train': int, 'val':int}
            epochs (int, optional): number of epochs to train. Defaults to 10.

        Returns:
            _type_: _description_
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size['train']
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters())
        
        metrics = getMetricsDict()
        metrics['loss'] = 0.0
        best_loss = float('inf')
        best_davg = 0.0
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in tqdm(range(epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.to(self.device).float()  # B, S, H, W, C
                    trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)  # B, N, T, 2
                    visibs_g = visibs_g.to(self.device)  # B, T, N

                    # Normalize trajectories
                    B, S, H, W, C = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1

                    # Select query points from random frames
                    if q_frame == -1 or phase == "val":
                        fix = int(S * 0.72)
                        #fix = 0 #just to finetune on RV 
                    elif q_frame == -2:
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix = q_frame
                    
                    q_points = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix).to(self.device)  # [B, N, 3]
                    q_points = q_points[..., [0, 2, 1]]  # Convert (t, x, y) -> (t, y, x)

                    with torch.set_grad_enabled(phase == 'train'):
                        outs = self.model(video=rgbs, query_points=q_points)
                        batch_loss = sequence_loss([outs['tracks']], trajs_g.permute(0, 2, 1, 3), visibs_g, visibs_g).mean()  # Mean across the batch

                        
                        if phase == 'train':
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            optimizer.step()
                            scheduler.step()

                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Evaluate predictions for each point
                        trajs_e = outs['tracks'].permute(0, 2, 1, 3)
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        q_points[..., 1] /= W - 1
                        q_points[..., 2] /= H - 1

                        outputs = evaluate.compute_metrics(
                            q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.cpu().numpy(),
                            first=False
                        )

                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)

                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] < best_loss:
                        saverloader.save(ckpt_path, optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] > best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()


class COTRACKER3():
    def __init__(
            self, 
            device_ids=list(range(torch.cuda.device_count())), 
            ft_model=False
        ) -> None:

        self.ft_model = ft_model
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60).to(self.device)

    def load(self, path=None, eval=True):
        """_summary_

        Args:
            path (_type_): _description_
            eval (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if path is not None and self.ft_model:
            _ = saverloader.load(path, self.model)
        elif path is not None:
            with open(path, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print(f"The CoTracker3 model is loaded for evaluation.")
            return self.model.eval().to(self.device)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print(f"The CoTracker3 model is loaded for training.")
            return self.model.train()

    @torch.no_grad()
    def infer(self, video, points, resize=None, add_support_grid=False, iters=6): #resize = (384, 512) from CoTrackerPredictor
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]
            points (tensor): a list of points across different timesteps [B, N, 3]->(t, x, y). range: [0.0 - 1.0]

        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]
            visibs_e (tensor): Boolean for each point on every frame [B, N, S]
        """
        rgbs = video.permute(0, 1, 4, 2, 3).float().to(self.device) # B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        if resize is not None:
            rgbs_ = rgbs.reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear', align_corners=True)
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W)
            _, S, C, H, W = rgbs.shape
        
        # converting to pixel level values
        points[...,1] *= W - 1
        points[...,2] *= H - 1
        _, N, _ = points.shape

        # from (t, x, y) to (t, y, x)
        #query_points = points[:,:,[0, 2, 1]] # (B, N, 3)
        query_points = points.to(self.device)


        if add_support_grid:
            support_grid_size = 6
            if resize is not None:
                grid_pts = get_points_on_a_grid(support_grid_size, resize, device=self.device)
            else:
                grid_pts = get_points_on_a_grid(support_grid_size, (H, W), device=self.device)
            
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            grid_pts = grid_pts.repeat(B, 1, 1)
            query_points = torch.cat([query_points, grid_pts], dim=1)
        

        trajs_e, visibs_e, confidence, __ = self.model(video=rgbs, queries=query_points, iters=iters)
        visibs_e = visibs_e * confidence
        
        if query_points[0,0,0].item() > 0.0: # backward tracking
            trajs_e, visibs_e = self._compute_backward_tracks(
                video=rgbs, queries=query_points, tracks=trajs_e, visibilities=visibs_e, iters=iters
            )
        
        if add_support_grid:
            trajs_e = trajs_e[:,:,:N]
            visibs_e = visibs_e[:,:,:N]
            confidence = confidence[:,:,:N]
        

        thr = 0.6
        visibs_e = visibs_e > thr

        #print(trajs_e.min(), trajs_e.max())
        #raise KeyboardInterrupt

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(query_points)):
            queries_t = query_points[i, :trajs_e.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            trajs_e[i, queries_t, arange] = query_points[i, :trajs_e.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibs_e[i, queries_t, arange] = True
        
        #print(trajs_e.min(), trajs_e.max())
        #raise KeyboardInterrupt
        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        
        points[...,1] /= W - 1
        points[...,2] /= H - 1

        return trajs_e.permute(0, 2, 1, 3).cpu(), visibs_e.permute(0, 2, 1).cpu()
    
    @torch.no_grad()
    def _compute_backward_tracks(self, video, queries, tracks, visibilities, iters):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1


        inv_tracks, inv_visibilities, confidence, __ = self.model(video=inv_video, queries=inv_queries, iters=iters)
        inv_visibilities = inv_visibilities * confidence
        

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        arange = torch.arange(video.shape[1], device=queries.device)[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities
    
    def finetune(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):
        """Train the model on the provided dataset.

        Args:
            dataloaders (Dict): {'train': torch train dataloader, 'val': torch validation dataloader}
            dataset_size (Dict): {'train': int, 'val': int}
            log_dir (str): Directory for TensorBoard logs.
            ckpt_path (str): Directory for saving checkpoints.
            epochs (int): Number of epochs to train. Defaults to 10.
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size["train"]
        optimizer, scheduler = fetch_optimizer(
            lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters()
        )
        
        # metrics = getMetricsDict()
        # metrics['loss'] = 0.0
        best_loss = float('inf')
        best_davg = 0.0

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in tqdm(range(epochs)):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.permute(0, 1, 4, 2, 3).float().to(self.device)  # [B, T, C, H, W]
                    trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)  # [B, N, T, 2]
                    visibs_g = visibs_g.permute(0, 2, 1).to(self.device)  # [B, T, N]
                    
                    B, S, C, H, W = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1

                    # Select query points from random frames
                    if q_frame == -1 or phase == "val":
                        fix = int(S * 0.75)
                    elif q_frame == -2:
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix = q_frame

                    #points_0 = trajs_g[:, :, q_frame, :]  # Initial points from q frame
                    q_points = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix).to(self.device)  # [B, N, 3]
                    # print(q_points[0,0,0])
                    
                    with torch.set_grad_enabled(phase == "train"):
                        trajs_e, visibs_e, confidence, train_data = self.model(video=rgbs, queries=q_points, is_train=True)
                        coord_predictions, _, _, _ = (train_data)                        
                        loss = sequence_loss(coord_predictions[-1], trajs_g, visibs_g, visibs_g, 0.8)
                        batch_loss = loss.mean()  # Mean across the batch (also DataParallel compatibility)

                        if phase == "train":
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                            optimizer.step()
                            scheduler.step()
                        
                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Normalize predictions and ground truth back to [0, 1]
                        #trajs_e = preds[-1]  # Final predicted trajectories

                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        # points_0[..., 0] /= W - 1
                        # points_0[..., 1] /= H - 1
                        q_points[..., 1] /= W - 1
                        q_points[..., 2] /= H - 1

                        # Evaluate metrics
                        outputs = evaluate.compute_metrics(
                            # points_0.cpu().numpy(),
                            q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            first=False
                        )
                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)  # Accumulate over batch size
                        
                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] < best_loss:
                        saverloader.save(ckpt_path, optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] > best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()


class PIPS2():
    def __init__(self, stride=8, device_ids=list(range(torch.cuda.device_count()))) -> None:
        """ initializing the pips++ architecture

        Args:
            stride (int, optional): spatial stride of the model. Defaults to 8.
            device_ids : a list of device ids, Defaults to [0]
        """
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model = Pips2(stride=stride).to(self.device)

    def load(self, path=None, eval=True):
        """_summary_

        Args:
            path (_type_): _description_
            eval (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if path is not None:
            _ = saverloader.load(path, self.model)
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print("The pips++ model is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print("The pips++ model is loaded for training.")
            return self.model.train()
        
    def infer(self, video, points, resize=None, q_frame=0): # (512, 896) is used in official repo
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]
            points (tensor): a list of points on first frame of the given video [B, N, 2]

        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]
        """
        rgbs = video.permute(0, 1, 4, 2, 3).float().to(self.device) # B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        if resize is not None:
            rgbs_ = rgbs.reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W)
            _, S, C, H, W = rgbs.shape
        # converting to pixel level values
        points[...,0] *= W - 1
        points[...,1] *= H - 1
        _, N, _ = points.shape

        trajs_e = points.unsqueeze(1).repeat(1,S,1,1).to(self.device)
        #print(trajs_e.shape, points.shape)
        
        with torch.no_grad():
            preds, _, _, _ = self.model(trajs_e, rgbs, iters=4, feat_init=None, beautify=True, q_frame=q_frame)
            #print(torch.cuda.memory_allocated()/ (1024 ** 2))
        
        trajs_e = preds[-1]
        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        points[...,0] /= W - 1
        points[...,1] /= H - 1
        trajs_e = trajs_e.cpu().permute(0, 2, 1, 3)
        return trajs_e#, occluded.squeeze()
    
    def train(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):        
        """ Finetune the model on the provided dataset.
        Args:
            dataloaders (Dict): {'train':torch train dataloader, 'val':torch train dataloader}
            dataset_size (Dict): {'train': int, 'val':int}
            epoch (int, optional): number of epochs to train. Defaults to 10.

        Returns:
            _type_: _description_
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size["train"]
        optimizer, scheduler = fetch_optimizer(
            lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters()
        )
        
        # metrics = getMetricsDict()
        # metrics['loss'] = 0.0
        # best_loss = 100000.0
        best_loss = float('inf')
        best_davg = 0.0
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in tqdm(range(epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                
                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.permute(0, 1, 4, 2, 3).float().to(self.device)  # [B, T, C, H, W]
                    trajs_g = trajs_g.permute(0, 2, 1, 3).float().to(self.device)  # [B, N, T, 2]
                    visibs_g = visibs_g.permute(0, 2, 1).to(self.device)  # [B, T, N]
                    
                    B, S, C, H, W = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1
                    
                    # Select query points from random frames
                    if q_frame == -1 or phase == "val":
                        fix = int(S * 0.75)
                    elif q_frame == -2:
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix = q_frame

                    points_0 = trajs_g[:,fix, :, :]  # Initial points from q frame
                    trajs_e = points_0.unsqueeze(1).repeat(1,S,1,1).to(self.device)# B, S, N, 2

                    with torch.set_grad_enabled(phase == "train"):
                        #print(trajs_e.shape, trajs_g.shape)
                        preds, _, _, loss = self.model(trajs_e, rgbs, iters=8, 
                                    trajs_g=trajs_g, vis_g=visibs_g, valids=visibs_g, beautify=True, is_train=True, q_frame=fix)
                        batch_loss = loss.mean()  # Mean across the batch (also DataParallel compatibility)

                        if phase == "train":
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                            optimizer.step()
                            scheduler.step()

                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Normalize predictions and ground truth back to [0, 1]
                        trajs_e = preds[-1]  # Final predicted trajectories

                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        points_0[..., 0] /= W - 1
                        points_0[..., 1] /= H - 1
                        

                        # Evaluate metrics

                        points_00 = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix)
                        #print(points_00.shape, trajs_g.shape)
                        outputs = evaluate.compute_metrics(
                            points_00.cpu().numpy(),
                            #q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            first=False
                        )
                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)  # Accumulate over batch size
                        
                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] < best_loss:
                        saverloader.save(ckpt_path, optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] > best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()


class ECHOTRACKER():
    def __init__(self, stride=8, device_ids=list(range(torch.cuda.device_count())),
                 feature_extractor_chunk_size=64,
        ) -> None:
        """ initializing the EchoTracker architecture

        Args:
            stride (int, optional): spatial stride of the model. Defaults to 8.
            device_ids : a list of device ids, Defaults to [0]
        """
        self.device = 'cuda:%d' % device_ids[0]
        self.device_ids = device_ids
        self.model = EchoTracker(stride=stride,
                feature_extractor_chunk_size=feature_extractor_chunk_size).to(self.device)

    def load(self, path=None, eval=True):
        """_summary_

        Args:
            path (_type_): _description_
            eval (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if path is not None:
            _ = saverloader.load(path, self.model)
            print(f"The EchoTracker model is loaded from {path}.")
        
        if eval:
            requires_grad(self.model.parameters(), False)
            print("The EchoTracker model is loaded for evaluation.")
            return self.model.eval()
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            requires_grad(self.model.parameters(), True)
            print("The EchoTracker model is loaded for training.")
            return self.model.train()
        
    def infer(self, video, points, resize=None):
        """Run the model on the video for the given points in 1st frame and return the results.

        Args:
            video (tensor): a sequence of frames of shape [B, S, H, W, C]. S must be divisible by 8. range: [0 - 255]
            points (tensor): a list of points across different timesteps [B, N, 3]->(t, x, y). range: [0.0 - 1.0]
            resize (tuple): (H, W)
        Returns:
            trajs_e (tensor): Estimated trajectory of points through the video [B, N, S, 2]. range: [0.0 - 1.0]
        """
        points = points.to(self.device)
        rgbs = video.permute(0, 1, 4, 2, 3).float().to(self.device) # B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        if resize is not None:
            rgbs_ = rgbs.reshape(B*S, C, H, W)
            H_, W_ = resize[0], resize[1]
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            H, W = H_, W_
            rgbs = rgbs_.reshape(B, S, C, H, W)
            _, S, C, H, W = rgbs.shape

        # converting to pixel level values
        if points.shape[-1] > 2:
            points[...,1] *= W - 1
            points[...,2] *= H - 1
        else:
            points[...,0] *= W - 1
            points[...,1] *= H - 1
        # _, N, _ = points.shape
        

        # print(f"in model: {rgbs.shape}, {rgbs.dtype}, {points.shape}")
        # raise KeyboardInterrupt
        with torch.no_grad():
            preds = self.model(rgbs, points, iters=4)
            #print(torch.cuda.memory_allocated()/ (1024 ** 2))
        
        trajs_e = preds[-1]
        
        trajs_e[...,0] /= W - 1
        trajs_e[...,1] /= H - 1
        if points.shape[-1] > 2:
            points[...,1] /= W - 1
            points[...,2] /= H - 1
        else:
            points[...,0] /= W - 1
            points[...,1] /= H - 1
        # points[...,0] /= W - 1
        # points[...,1] /= H - 1
        trajs_e = trajs_e.cpu().permute(0, 2, 1, 3)
        return trajs_e
    
    def train(self, dataloaders, dataset_size, log_dir, ckpt_path, epochs=10, q_frame=0):
        """Train the model on the provided dataset.

        Args:
            dataloaders (Dict): {'train': torch train dataloader, 'val': torch validation dataloader}
            dataset_size (Dict): {'train': int, 'val': int}
            log_dir (str): Directory for TensorBoard logs.
            ckpt_path (str): Directory for saving checkpoints.
            epochs (int): Number of epochs to train. Defaults to 10.
        """
        lr = 5e-4
        weight_decay = 1e-6
        steps_per_epoch = dataset_size["train"]
        optimizer, scheduler = fetch_optimizer(
            lr, weight_decay, 1e-8, epochs, steps_per_epoch, self.model.parameters()
        )
        
        # metrics = getMetricsDict()
        # metrics['loss'] = 0.0
        best_loss = float('inf')
        best_davg = 0.0

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in tqdm(range(epochs)):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                # Reset batch metrics
                epoch_metrics = getMetricsDict()
                epoch_metrics['loss'] = 0.0
                total_batches = 0  # To keep track of batches in each phase

                for rgbs, trajs_g, visibs_g in dataloaders[phase]:
                    rgbs = rgbs.permute(0, 1, 4, 2, 3).float().to(self.device)  # [B, T, C, H, W]
                    trajs_g = trajs_g.permute(0, 2, 1, 3).to(self.device)  # [B, N, T, 2]
                    visibs_g = visibs_g.permute(0, 2, 1).to(self.device)  # [B, T, N]
                    
                    B, S, C, H, W = rgbs.shape
                    trajs_g[..., 0] *= W - 1
                    trajs_g[..., 1] *= H - 1

                    # Select query points from random frames
                    if q_frame == -1:
                        fix = int(S * 0.75)
                    elif q_frame == -2:
                        fix = torch.randint(0, S, (1,)).item()
                    else:
                        fix=q_frame

                    #points_0 = trajs_g[:, :, q_frame, :]  # Initial points from q frame
                    q_points = randomize_points(trajs_g.permute(0, 2, 1, 3), fix=fix).to(self.device)  # [B, N, 3]
                    # print(q_points[0,0,0])
                    
                    with torch.set_grad_enabled(phase == "train"):
                        preds = self.model(rgbs=rgbs, points_0=q_points)
                        batch_loss = sequence_loss(preds, trajs_g, visibs_g, visibs_g, 0.8).mean()  # Mean across the batch (also DataParallel compatibility)

                        if phase == "train":
                            optimizer.zero_grad()
                            batch_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                            optimizer.step()
                            scheduler.step()
                        
                        # Accumulate metrics and loss
                        epoch_metrics['loss'] += batch_loss.item() * rgbs.size(0)
                        total_batches += rgbs.size(0)

                        # Normalize predictions and ground truth back to [0, 1]
                        trajs_e = preds[-1]  # Final predicted trajectories

                        trajs_g[..., 0] /= W - 1
                        trajs_g[..., 1] /= H - 1
                        trajs_e[..., 0] /= W - 1
                        trajs_e[..., 1] /= H - 1
                        # points_0[..., 0] /= W - 1
                        # points_0[..., 1] /= H - 1
                        q_points[..., 1] /= W - 1
                        q_points[..., 2] /= H - 1

                        # Evaluate metrics
                        outputs = evaluate.compute_metrics(
                            # points_0.cpu().numpy(),
                            q_points.cpu().numpy(),
                            trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            trajs_e.detach().permute(0, 2, 1, 3).cpu().numpy(),
                            visibs_g.permute(0, 2, 1).cpu().numpy(),
                            first=False
                        )
                        for key, value in outputs.items():
                            epoch_metrics[key] += value * rgbs.size(0)  # Accumulate over batch size
                        
                # Normalize metrics by dataset size
                for key in epoch_metrics.keys():
                    epoch_metrics[key] /= total_batches

                # Logging and saving models
                if phase == 'val':
                    saverloader.save(os.path.join(ckpt_path, 'lasts'), optimizer, self.model.module, epoch, scheduler=scheduler)
                    if epoch_metrics['loss'] < best_loss:
                        saverloader.save(ckpt_path, optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_loss = epoch_metrics['loss']
                    if epoch_metrics['average_pts_within_thresh'] > best_davg:
                        #saving best model based on average position accuracy
                        saverloader.save(os.path.join(ckpt_path, 'best_davg') , optimizer, self.model.module, epoch, scheduler=scheduler)
                        best_davg = epoch_metrics['average_pts_within_thresh']

                print(f"Epoch: {epoch} -> {phase}: loss={epoch_metrics['loss']:.4f}, d_avg={epoch_metrics['average_pts_within_thresh']:.4f}")
                writer.add_scalars(f"{phase}", epoch_metrics, epoch)
                writer.flush()

        writer.close()


