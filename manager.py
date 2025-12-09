import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import utils.data_loaders

import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils_clamp import get_loss_clamp, get_loss_mvp
from utils.loss_utils import get_ae_loss, get_denoise_loss
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from utils.mvp_utils import *
from PIL import Image

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, extend1, cfg, extend2=None):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Create the optimizers
        # Primary optimizer now jointly updates:
        #   - primary completion network parameters (model)
        #   - auxiliary heads (extend1, extend2) so that meta-loss gradients
        #     can align auxiliary behavior with the primary objective.
        main_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        aux_params = list(filter(lambda p: p.requires_grad, extend1.parameters()))
        self.extend2 = extend2
        if self.extend2 is not None:
            aux_params += list(filter(lambda p: p.requires_grad, self.extend2.parameters()))
        all_params = main_params + aux_params

        self.optimizer = torch.optim.Adam(all_params,
                                          lr=cfg.TRAIN.LEARNING_RATE,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                          betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # self.lr_decays = {i: 0.1 ** (1 / cfg.TRAIN.LR_DECAY) for i in range(1, cfg.TRAIN.N_EPOCHS+1)}
        # self.scheduler_steplr = StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=self.scheduler_steplr)
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # Meta-learned auxiliary weighting (λ_smr, λ_ad) and weight on aux losses
        # create λ on the same device as the model parameters
        param_device = next(model.parameters()).device
        self.lambda_smr = torch.tensor(0.0, device=param_device, requires_grad=True)
        self.lambda_ad = torch.tensor(0.0, device=param_device, requires_grad=True)
        lambda_lr = getattr(cfg.TRAIN, 'LAMBDA_LR', 1e-3)
        self.lambda_optimizer = torch.optim.Adam([self.lambda_smr, self.lambda_ad], lr=lambda_lr)
        # global scalar that balances primary vs. auxiliary objectives
        self.meta_weight = getattr(cfg.TRAIN, 'META_WEIGHT', 1.0)
        ##########################
        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, extend1,train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()
            extend1.train()

            # Update learning rate
            self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0
            #### auxiliary
            ax_cd = 0
            ax2_cd = 0
            #############
            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # # unpack data
                # partial = data['partial_cloud']
                # gt = data['gtcloud']
                partial = data[0].cuda()
                gt = data[1].cuda()
      
                # ------------------------------
                # (i) Inner: multi-sample auxiliary averaging (no optimizer.step)
                # ------------------------------
                inner_steps = 3
                device = partial.device
                loss_aux1_acc = torch.tensor(0.0, device=device)
                loss_aux2_acc = torch.tensor(0.0, device=device)

                for _ in range(inner_steps):
                    bsz = partial.size(0)
                    inner_idx = np.random.randint(0, bsz)
                    inner_partial = partial[inner_idx:inner_idx + 1]
                    inner_gt = gt[inner_idx:inner_idx + 1]

                    # Aux 1: MAE
                    rec = extend1(inner_partial)
                    la1, _ = get_ae_loss(rec, inner_partial)
                    loss_aux1_acc += la1

                    # Aux 2: denoising (optional)
                    if self.extend2 is not None:
                        offset = self.extend2(inner_partial)
                        la2, _ = get_denoise_loss(offset, inner_partial, inner_gt, sqrt=True)
                        loss_aux2_acc += la2

                loss_aux1 = loss_aux1_acc / float(inner_steps)
                loss_aux2 = (loss_aux2_acc / float(inner_steps)) if self.extend2 is not None else torch.tensor(0.0, device=device)

                # accumulate aux metrics for logging (use averaged losses)
                ax_cd += loss_aux1.item() * 1e2
                if self.extend2 is not None:
                    ax2_cd += loss_aux2.item() * 1e2

                # ------------------------------
                # (ii) Meta-learned weighting λ
                # ------------------------------
                alpha_tilde = torch.log(1 + self.lambda_smr ** 2)
                beta_tilde = torch.log(1 + self.lambda_ad ** 2)
                w_smr = torch.exp(alpha_tilde) / (torch.exp(alpha_tilde) + torch.exp(beta_tilde))
                w_ad = 1.0 - w_smr

                loss_aux_total = w_smr * loss_aux1 + w_ad * loss_aux2

                # ------------------------------
                # (iii) Outer: Main task + meta-weighted aux
                # ------------------------------
                pcds_pred = model(partial)
                loss_main, losses, gts = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                meta_loss = loss_main + self.meta_weight * loss_aux_total

                self.optimizer.zero_grad()
                self.lambda_optimizer.zero_grad()
                meta_loss.backward()
                self.optimizer.step()
                self.lambda_optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item

                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

                # if steps <= cfg.TRAIN.WARMUP_STEPS:
                #     self.lr_scheduler.step()
                #     steps += 1

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches
            ### aux
            avg_ae = ax_cd / n_batches
            avg_denoise = ax2_cd / n_batches if self.extend2 is not None else 0

            # Update learning rate
            # if self.epoch in self.lr_decays:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] *= self.lr_decays[self.epoch]
            # else:
            #     raise ValueError('Epoch exceeds max limit!')
            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial, avg_ae, avg_denoise]]))

            # Validate the current model
            cd_eval = self.validate(cfg, model=model, extend1=extend1,val_data_loader=val_data_loader)

            # Save checkpoints
            if cd_eval < self.best_metrics:
                if cd_eval < self.best_metrics:
                    self.best_epoch = epoch_idx
                    file_name = 'ckpt-best.pth'
                    extendM1_name="extend-ckpt-best.pth" 
                
                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                    extendM1_name="extend-ckpt-epoch-%03d.pth" % epoch_idx
                
                
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                output_path_e1 = os.path.join(cfg.DIR.CHECKPOINTS, extendM1_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': extend1.state_dict()
                }, output_path_e1)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None, extend1=None, val_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()
        extend1.eval()
        if self.extend2 is not None:
            self.extend2.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())

        ######## aux
        ax_loss=AverageMeter(['ae'])
        ax_metrics = AverageMeter(Metrics.names())
        denoise_loss = AverageMeter(['denoise'])

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                # partial = data['partial_cloud']
                # gt = data['gtcloud']
                partial=data[0].cuda()
                gt=data[1].cuda()

                for _ in range(5):
                    rec = extend1(partial)
                    loss_all,loss_ax = get_ae_loss(rec,partial)
                    loss_ae = loss_ax[0].item()*1e2
                    a_metrics = [loss_ae]
                    ax_loss.update([loss_ae])
                    ax_metrics.update(a_metrics)

                if self.extend2 is not None:
                    for _ in range(5):
                        offset = self.extend2(partial)
                        loss_all2, loss_ax2 = get_denoise_loss(offset, partial, gt, sqrt=True)
                        loss_dn = loss_ax2[0].item() * 1e2
                        denoise_loss.update([loss_dn])
                

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)

        return test_losses.avg(3)


    def test(self, cfg, model=None, test_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()
        category_models = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # partial = data['partial_cloud']
                # gt = data['gtcloud']
                partial=data[0].cuda()
                gt=data[1].cuda()

                b, n, _ = partial.shape

                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)


                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # record
                        if taxonomy_id not in category_models:
                            category_models[taxonomy_id] = []
                        category_models[taxonomy_id].append((model_name, cd3))
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category model cds
        if outdir:
            for cat in list(category_models.keys()):
                model_tuples = category_models[cat]
                with open(os.path.join(outdir, cat+'.txt'), 'w') as record_file:
                    # sort by cd
                    for tt in sorted(model_tuples, key=lambda x:x[1]):
                        record_file.write('{:s}  {:.4f}\n'.format(tt[0], tt[1])) 

        # # Print testing results
        # print('============================ TEST RESULTS ============================')
        # print('Taxonomy', end='\t')
        # print('#Sample', end='\t')
        # for metric in test_metrics.items:
        #     print(metric, end='\t')
        # print()

        # for taxonomy_id in category_metrics:
        #     print(taxonomy_id, end='\t')
        #     print(category_metrics[taxonomy_id].count(0), end='\t')
        #     for value in category_metrics[taxonomy_id].avg():
        #         print('%.4f' % value, end='\t')
        #     print()

        # print('Overall', end='\t\t\t')
        # for value in test_metrics.avg():
        #     print('%.4f' % value, end='\t')
        # print('\n')

        # print('Epoch ', self.epoch, end='\t')
        # for value in test_losses.avg():
        #     print('%.4f' % value, end='\t')
        # print('\n')

        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            self.train_record(message)

        self.train_record('Overall\t\t' + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)


class Manager_shapenet55:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model,extend1, cfg, extend2=None):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # training dataset
        self.dataset = cfg.DATASET.TRAIN_DATASET

        # Epoch index
        self.epoch = 0

        # Create the optimizers
        # Joint optimizer over primary completion model and auxiliary heads
        main_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        aux_params = list(filter(lambda p: p.requires_grad, extend1.parameters()))
        self.extend2 = extend2
        if self.extend2 is not None:
            aux_params += list(filter(lambda p: p.requires_grad, self.extend2.parameters()))
        all_params = main_params + aux_params

        self.optimizer = torch.optim.Adam(all_params,
                                          lr=cfg.TRAIN.LEARNING_RATE,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                          betas=cfg.TRAIN.BETAS)

        # lr scheduler
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # Meta-learned auxiliary weighting (λ_smr, λ_ad) and weight on aux losses
        param_device = next(model.parameters()).device
        self.lambda_smr = torch.tensor(0.0, device=param_device, requires_grad=True)
        self.lambda_ad = torch.tensor(0.0, device=param_device, requires_grad=True)
        lambda_lr = getattr(cfg.TRAIN, 'LAMBDA_LR', 1e-3)
        self.lambda_optimizer = torch.optim.Adam([self.lambda_smr, self.lambda_ad], lr=lambda_lr)
        self.meta_weight = getattr(cfg.TRAIN, 'META_WEIGHT', 1.0)
        ##########################

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()

    def unpack_data(self, data):

        if self.dataset == 'PCN':
            partial = data[0]
            gt =data[1]
        elif self.dataset == 'ShapeNet':
            partial = data['partial_cloud']
            gt = data['gtcloud']
        elif self.dataset == 'ShapeNet55':
            # generate partial data online
            gt = data.cuda()
            _, npoints, _ = gt.shape
            partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        else:
            raise ValueError('No method implemented for this dataset: {:s}'.format(self.dataset))

        return partial, gt


    def train(self, model,extend1, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()
            extend1.train()

            # Update learning rate
            self.lr_scheduler.step()
        

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0
            #### auxiliary
            ax_cd = 0
            ax2_cd = 0
            #############

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)
                
                # ------------------------------
                # (i) Inner: multi-sample auxiliary averaging (no optimizer.step)
                # ------------------------------
                inner_steps = 3
                device = partial.device
                loss_aux1_acc = torch.tensor(0.0, device=device)
                loss_aux2_acc = torch.tensor(0.0, device=device)

                for _ in range(inner_steps):
                    bsz = partial.size(0)
                    inner_idx = np.random.randint(0, bsz)
                    inner_partial = partial[inner_idx:inner_idx + 1]
                    inner_gt = gt[inner_idx:inner_idx + 1]

                    # Aux 1: MAE
                    rec = extend1(inner_partial)
                    la1, _ = get_ae_loss(rec, inner_partial)
                    loss_aux1_acc += la1

                    # Aux 2: denoising
                    if self.extend2 is not None:
                        offset = self.extend2(inner_partial)
                        la2, _ = get_denoise_loss(offset, inner_partial, inner_gt, sqrt=False)
                        loss_aux2_acc += la2

                loss_aux1 = loss_aux1_acc / float(inner_steps)
                loss_aux2 = (loss_aux2_acc / float(inner_steps)) if self.extend2 is not None else torch.tensor(0.0, device=device)

                # accumulate aux metrics for logging (use averaged losses)
                ax_cd += loss_aux1.item() * 1e2
                if self.extend2 is not None:
                    ax2_cd += loss_aux2.item() * 1e2
              
                # ------------------------------
                # (ii) Meta-learned weighting λ
                # ------------------------------
                alpha_tilde = torch.log(1 + self.lambda_smr ** 2)
                beta_tilde = torch.log(1 + self.lambda_ad ** 2)
                w_smr = torch.exp(alpha_tilde) / (torch.exp(alpha_tilde) + torch.exp(beta_tilde))
                w_ad = 1.0 - w_smr

                loss_aux_total = w_smr * loss_aux1 + w_ad * loss_aux2

                # ---------------------------
                # (iii) Outer: Main task + meta-weighted aux
                # ---------------------------
                pcds_pred = model(partial)

                loss_main, losses, gts = get_loss_clamp(pcds_pred, partial, gt, sqrt=False) ## false == L2
                meta_loss = loss_main + self.meta_weight * loss_aux_total

                self.optimizer.zero_grad()
                self.lambda_optimizer.zero_grad()
                meta_loss.backward()
                self.optimizer.step()
                self.lambda_optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            ### aux
            avg_ae= ax_cd / n_batches
            avg_denoise = ax2_cd / n_batches if self.extend2 is not None else 0

            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial,avg_ae,avg_denoise]]))

            # Validate the current model
            cd_eval = self.validate(cfg, model=model, extend1=extend1,val_data_loader=val_data_loader)
            self.train_record('Testing scores = {:.4f}'.format(cd_eval))

            # Save checkpoints
            if cd_eval < self.best_metrics:
                if cd_eval < self.best_metrics:
                    self.best_epoch = epoch_idx
                    file_name = 'ckpt-best.pth'
                    extendM1_name="extend-ckpt-best.pth" 
                
                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                    extendM1_name="extend-ckpt-epoch-%03d.pth" % epoch_idx
                
                
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                output_path_e1 = os.path.join(cfg.DIR.CHECKPOINTS, extendM1_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': extend1.state_dict()
                }, output_path_e1)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None,  extend1=None, val_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()
        extend1.eval()
        if self.extend2 is not None:
            self.extend2.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter('cd3')

        ######## aux
        ax_loss=AverageMeter(['ae'])
        ax_metrics = AverageMeter(Metrics.names())
        denoise_loss = AverageMeter(['denoise'])

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                for _ in range(3):
                    rec = extend1(partial)
                    loss_all,loss_ax = get_ae_loss(rec,partial)
                    loss_ae = loss_ax[0].item()*1e2
                    a_metrics = [loss_ae]
                    ax_loss.update([loss_ae])
                    ax_metrics.update(a_metrics)

                if self.extend2 is not None:
                    for _ in range(3):
                        offset = self.extend2(partial)
                        loss_all2, loss_ax2 = get_denoise_loss(offset, partial, gt, sqrt=False)
                        loss_dn = loss_ax2[0].item() * 1e2
                        denoise_loss.update([loss_dn])

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=False)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message, show_info=False)

        return test_losses.avg(3)

    def test(self, cfg, model,extend1, test_data_loader, outdir, mode=None):

        if self.dataset == 'ShapeNet':
            self.test_pcn(cfg, model, test_data_loader, outdir)
        elif self.dataset == 'ShapeNet55':
            self.test_shapenet55(cfg, model, extend1,test_data_loader, outdir, mode)
        else:
            raise ValueError('No testing method implemented for this dataset: {:s}'.format(self.dataset))

    def test_pcn(self, cfg, model=None, test_data_loader=None, outdir=None):
        """
        Testing Method for dataset PCN
        """

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                for _ in range(5):
                    rec=extend1(partial)
                    loss_all,loss_ax=get_ae_loss(rec,partial)
                    loss_ae=loss_ax[0].item()*1e2
                    a_metrics=[loss_ae]
                    ax_loss.update([loss_ae])
                    ax_metrics.update(a_metrics)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=True)

                # get loss
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                # get all metrics
                _metrics = Metrics.get(pcds_pred[-1], gt)
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)

    def test_shapenet55(self, cfg, model=None, extend1=None, test_data_loader=None, outdir=None, mode=None):
        """
        Testing Method for dataset shapenet-55/34
        """

        from models.utils import fps_subsample
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Eval settings
        crop_ratio = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }
        choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]

        # Switch models to evaluation mode
        model.eval()
        extend1.eval()

        ######## aux
        ax_loss=AverageMeter(['ae'])
        ax_metrics = AverageMeter(Metrics.names())

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                partial, gt = self.unpack_data(data)
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)

                    for _ in range(10):
                        rec=extend1(partial)
                        loss_all,loss_ax=get_ae_loss(rec,partial)
                        loss_ae=loss_ax[0].item()*1e2
                        a_metrics=[loss_ae]
                        ax_loss.update([loss_ae])
                        ax_metrics.update(a_metrics)

                    pcds_pred = model(partial.contiguous())
                    loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=False) # L2

                    # get loss
                    cdc = losses[0].item() * 1e3
                    cd1 = losses[1].item() * 1e3
                    cd2 = losses[2].item() * 1e3
                    cd3 = losses[3].item() * 1e3
                    partial_matching = losses[4].item() * 1e3
                    test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                    # get all metrics
                    _metrics = Metrics.get(pcds_pred[-1], gt)
                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)

                    # output to file
                    if outdir:
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                        # save pred, gt, partial pcds 
                        pred = pcds_pred[-1]
                        for mm, model_name in enumerate(model_id):
                            output_file = os.path.join(outdir, taxonomy_id, model_name+'_{:02d}'.format(partial_id))
                            write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            # output img files
                            img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                            output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                            output_img = (output_img*255).astype('uint8')
                            im = Image.fromarray(output_img)
                            im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)


class Manager_MVP:
    def __init__(self, model, extend1, cfg, extend2=None):


        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Joint optimizer over primary completion model and auxiliary heads
        main_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        aux_params = list(filter(lambda p: p.requires_grad, extend1.parameters()))
        self.extend2 = extend2
        if self.extend2 is not None:
            aux_params += list(filter(lambda p: p.requires_grad, self.extend2.parameters()))
        all_params = main_params + aux_params

        self.optimizer = torch.optim.Adam(all_params,
                                          lr=cfg.TRAIN.LEARNING_RATE,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                          betas=cfg.TRAIN.BETAS)

        # lr scheduler
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # Meta-learned auxiliary weighting (λ_smr, λ_ad) and weight on aux losses
        param_device = next(model.parameters()).device
        self.lambda_smr = torch.tensor(0.0, device=param_device, requires_grad=True)
        self.lambda_ad = torch.tensor(0.0, device=param_device, requires_grad=True)
        lambda_lr = getattr(cfg.TRAIN, 'LAMBDA_LR', 1e-3)
        self.lambda_optimizer = torch.optim.Adam([self.lambda_smr, self.lambda_ad], lr=lambda_lr)
        self.meta_weight = getattr(cfg.TRAIN, 'META_WEIGHT', 1.0)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, extend1, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')
        cd_best  = 1e6
        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()
            extend1.train()

            # Update learning rate
            self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0
            # auxiliary metrics (averaged aux losses)
            ax_cd = 0
            ax2_cd = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (label, partial, gt) in enumerate(train_data_loader):
                # unpack data
                partial = partial.float().cuda()
                gt = gt.float().cuda()

                # ------------------------------
                # (i) Inner: multi-sample auxiliary averaging (no optimizer.step)
                # ------------------------------
                inner_steps = 3
                device = partial.device
                loss_aux1_acc = torch.tensor(0.0, device=device)
                loss_aux2_acc = torch.tensor(0.0, device=device)

                for _ in range(inner_steps):
                    bsz = partial.size(0)
                    inner_idx = np.random.randint(0, bsz)
                    inner_partial = partial[inner_idx:inner_idx + 1]
                    inner_gt = gt[inner_idx:inner_idx + 1]

                    # Aux 1: MAE (using partial as input cloud)
                    rec = extend1(inner_partial)
                    la1, _ = get_ae_loss(rec, inner_partial)
                    loss_aux1_acc += la1

                    # Aux 2: denoising (optional)
                    if self.extend2 is not None:
                        offset = self.extend2(inner_partial)
                        la2, _ = get_denoise_loss(offset, inner_partial, inner_gt, sqrt=True)
                        loss_aux2_acc += la2

                loss_aux1 = loss_aux1_acc / float(inner_steps)
                loss_aux2 = (loss_aux2_acc / float(inner_steps)) if self.extend2 is not None else torch.tensor(0.0, device=device)

                # accumulate aux metrics for logging (use averaged losses)
                ax_cd += loss_aux1.item() * 1e2
                if self.extend2 is not None:
                    ax2_cd += loss_aux2.item() * 1e2

                # ------------------------------
                # (ii) Meta-learned weighting λ
                # ------------------------------
                alpha_tilde = torch.log(1 + self.lambda_smr ** 2)
                beta_tilde = torch.log(1 + self.lambda_ad ** 2)
                w_smr = torch.exp(alpha_tilde) / (torch.exp(alpha_tilde) + torch.exp(beta_tilde))
                w_ad = 1.0 - w_smr

                loss_aux_total = w_smr * loss_aux1 + w_ad * loss_aux2

                # ---------------------------
                # (iii) Outer: Main task + meta-weighted aux
                # ---------------------------
                pcds_pred = model(partial)

                loss_main, losses, gts = get_loss_mvp(pcds_pred, partial, gt, sqrt=True)
                meta_loss = loss_main + self.meta_weight * loss_aux_total

                self.optimizer.zero_grad()
                self.lambda_optimizer.zero_grad()
                meta_loss.backward()
                self.optimizer.step()
                self.lambda_optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                    n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # save checkpoint based on validation cd_t
            now_cd = self.validate(cfg, model, val_data_loader)

            if now_cd < cd_best:
                cd_best = now_cd
                file_name = 'ckpt-best.pth' 
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'model': model.state_dict()
                }, output_path)
            
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) '
                'Losses = %s cd_best = %f aux = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate,
                 epoch_end_time - epoch_start_time,
                 ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]],
                 cd_best,
                 ['%.4f' % l for l in [ax_cd / n_batches,
                                       (ax2_cd / n_batches) if self.extend2 is not None else 0.0]]))
        # training end
        self.train_record_file.close()
        self.test_record_file.close()

    def validate(self, cfg, model=None, val_data_loader=None):


        metrics = ['cd_p', 'cd_t', 'f1']
        test_loss_meters = {m: AverageValueMeter() for m in metrics}
        model.eval()

        with torch.no_grad():
            for data in val_data_loader:
                label, inputs_cpu, gt_cpu = data

                inputs = inputs_cpu.float().cuda()
                gt = gt_cpu.float().cuda()
                
                output = model(inputs)[-1]
                cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)
                result_dict = dict()
                result_dict['cd_p'] = cd_p
                result_dict['cd_t'] = cd_t
                result_dict['f1'] = f1

                for k, v in test_loss_meters.items():
                    v.update(result_dict[k].mean().item())
                
            
        return test_loss_meters['cd_t'].avg