import argparse
import os.path as osp
import time

import numpy as np


import os
import sys
import socket

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg, make_root_dir
from model import create_model
from loss import Evaluator
import torch

from geotransformer.utils.diffusion import Diffusion


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')

    parser.add_argument('--iteration', default=0,type=int, help='the iteration number for each model')

    parser.add_argument('--snapshot', type=str, default=None, help='load from snapshot')

    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')

    parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    
    parser.add_argument('--one_way_cross_type', type = str, default='non_to_anchor', 
                        help="""choose from [
                                non_to_anchor
                                all_to_anchor,
                                None] 
                            """) 
    
    parser.add_argument('--fix_train_T', action="store_true", 
        help='use a fixed time stamp T in training, ins.of random sampling')
    
    parser.add_argument('--train_preset_T', type=int, default=999, 
        help='always use the same T for training, if --fix_train_T is set')

    parser.add_argument('--use_pre_step', action='store_true',
        help='use pre step as input') 

    parser.add_argument('--note', type=str, default="",
        help='if you want to add any notes to the checkpoint folder')
    
    parser.add_argument('--ckpt_path', type=str, default=None,
        help='path to checkpint to load')    
    
    parser.add_argument('--position_encoding', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--pe_selfoneway', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--pe_crossoneway', action="store_true", 
        help='use position encoding for the features')
    
    parser.add_argument('--Slerp', action="store_true", 
        help='use Slerp to perform rotation addition')
    
    parser.add_argument('--ddim', action="store_true", 
        help='use ddim in reverse process')
    
    return parser

def x0_to_xt(time_step, data_dict, pred_x0, diffusion, engine_args):
    time_params = {}
    time_params["time_step"] = time_step.cuda().float()

    pred_x0_tmp = pred_x0.clone()
    prior = data_dict['estimated_transform'].clone()
    if engine_args.Slerp:
            noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat = \
            diffusion.mix_prior_with_GT_Slerp(pred_x0_tmp, \
                    prior, time_step)
    else:
        noised_transform, sqrt_alpha_hat, sqrt_one_minus_alpha_hat = \
            diffusion.mix_prior_with_GT(pred_x0_tmp, \
                    prior, time_step)

    
    data_dict["estimated_transform"] = noised_transform
    return data_dict

class Tester(SingleTester):
    def __init__(self, cfg):
        parser=make_parser()
        parsed_args = parser.parse_args()
        cfg = make_root_dir(parsed_args)
        self.cfg = cfg
        super().__init__(cfg, parser=parser)
        self.parsed_args = parsed_args
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark,iteration=self.args.iteration)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg, parsed_args=parsed_args).cuda()
        self.parsed_args = parsed_args
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        if self.parsed_args.use_pre_step:
            scene_name = data_dict['scene_name']
            ref_id = data_dict['ref_frame']
            src_id = data_dict['src_frame']
            dir = osp.join(self.cfg.feature_dir, self.args.benchmark)
            file_name = osp.join(dir, scene_name, f'{ref_id}_{src_id}.npz')
            pre_data_dict = np.load(file_name)
            if self.parsed_args.ddim:
                diffusion = Diffusion()
                time_step = torch.Tensor([self.parsed_args.train_preset_T]).int()    
                data_dict = x0_to_xt(time_step, data_dict, torch.from_numpy(pre_data_dict['estimated_transform']).cuda(), diffusion, self.parsed_args)
            else:
                data_dict['estimated_transform'] = torch.from_numpy(pre_data_dict['estimated_transform']).cuda()            
            
        time_step = torch.Tensor([self.parsed_args.train_preset_T]).cuda().float()
        data_dict["time_step"] = time_step

        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
