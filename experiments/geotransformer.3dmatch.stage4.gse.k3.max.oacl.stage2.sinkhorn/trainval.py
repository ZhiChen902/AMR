import argparse
import time

import torch
import torch.optim as optim
import os

import sys
import torch

import socket
if "yr-cuda11.3" in socket.gethostname():
    pass
elif "Zhifei-PC" in socket.gethostname():
    pass
else:
    sys.path.remove("/mnt/F/chenzhi/DiffReg-peal")
sys.path.append(os.getcwd())

print(sys.path)

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg, make_root_dir
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')

    parser.add_argument('--iteration', default=0,type=int, help='test benchmark')
    
    parser.add_argument('--one_way_cross_type', type = str, default='non_to_anchor', 
                        help="""choose from [
                                non_to_anchor
                                all_to_anchor,
                                None] 
                            """) 

    parser.add_argument('--note', type=str, default="",
        help='if you want to add any notes to the checkpoint folder')
    
    parser.add_argument('--ckpt_path', type=str, default=None,
        help='path to checkpint to load')   
    
    parser.add_argument('--fix_train_T', action="store_true", 
        help='use a fixed time stamp T in training, ins.of random sampling')
    
    parser.add_argument('--train_preset_T', type=int, default=1, 
        help='always use the same T for training, if --fix_train_T is set')
    
    parser.add_argument('--position_encoding', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--pe_selfoneway', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--pe_crossoneway', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--Slerp', action="store_true", 
        help='use Slerp to perform rotation addition')
    return parser

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        parser = make_parser()
        parsed_args = parser.parse_args()
        cfg = make_root_dir(parsed_args)

        super().__init__(cfg, max_epoch=cfg.optim.max_epoch, parser=parser, parsed_args= parsed_args)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg, parsed_args=parsed_args).cuda()
        model = self.register_model(model)


        # pretrained model 
        if parsed_args.ckpt_path is not None:
            pretrained_dict = torch.load(parsed_args.ckpt_path)['model']
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict  and 'pre_overlap' not in k  and 'ref_overlap' not in k and 'src_overlap' not in k)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print("pretrain - model", pretrained_dict.keys() - model_dict.keys())
            print("model - pretrain", model_dict.keys() - pretrained_dict.keys())


        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)

        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
