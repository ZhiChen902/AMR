import os
import os.path as osp
from typing import Tuple, Dict

import ipdb
import torch
import tqdm
import numpy as np

from geotransformer.engine.base_trainer import BaseTrainer
from geotransformer.utils.torch import to_cuda
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string

from geotransformer.utils.diffusion import Diffusion


def prepare_one_data(time_step, data_dict, diffusion, engine_args):
    time_params = {}
    time_params["time_step"] = time_step.cuda().float()
    # import pdb; pdb.set_trace()
    
    if engine_args.Slerp:
        noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat = \
        diffusion.mix_prior_with_GT_Slerp(data_dict['transform'].clone(), \
                data_dict['estimated_transform'].clone(), time_step)
    else:
        noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat = \
            diffusion.mix_prior_with_GT(data_dict['transform'].clone(), \
                    data_dict['estimated_transform'].clone(), time_step)

    time_params["sqrt_alpha_hat"] = sqrt_alpha_hat.detach()
    time_params["sqrt_one_minus_alpha_hat"] = sqrt_one_minus_alpha_hat.detach()
    data_dict["estimated_transform"] = noised_transform
    data_dict["time_params"] = time_params
    data_dict["time_step"] = time_step.cuda().float()

    return data_dict


class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
        parsed_args=None
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.cfg = cfg
        self.max_epoch = max_epoch
        self.parsed_args = parsed_args

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()

    def train_epoch(self):
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)

        diffusion = Diffusion()
        for iteration, data_dict in enumerate(self.train_loader):
            if(not data_dict["datavlid"]):
                continue
            self.inner_iteration = iteration + 1
            self.iteration += 1
            data_dict = to_cuda(data_dict)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()
            
            
            if self.parsed_args.fix_train_T:
                # use fixed T for training
                time_step = torch.Tensor([self.parsed_args.train_preset_T]).long()
            else:
                time_step = diffusion.sample_timesteps(self.cfg.train.batch_size).long()
            # import pdb; pdb.set_trace()
            data_dict = prepare_one_data(time_step, data_dict, diffusion, self.parsed_args)

            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
            
            # backward & optimization
            result_dict['loss'].backward()
            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.inner_iteration)
            # after training
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            torch.cuda.empty_cache()
        self.after_train_epoch(self.epoch)
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.critical(message)
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = f'epoch-{self.epoch - 1}.pth.tar'
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            print(iteration)
            data_dict = to_cuda(data_dict)
            data_dict["time_step"] = torch.Tensor([999]).cuda()
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            summary_dict = summary_board.summary()
            self.write_event('val-iter', summary_dict, iteration*self.epoch+iteration)
            pbar.set_description(message)
            torch.cuda.empty_cache()

        self.after_val_epoch(self.epoch)
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()

        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()
