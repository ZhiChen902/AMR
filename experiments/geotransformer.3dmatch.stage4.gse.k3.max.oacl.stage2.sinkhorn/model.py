import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import numpy as np
from geotransformer.modules.ops import pairwise_distance
from geotransformer.datasets.registration.threedmatch.dataset import get_correspondences
import numpy as np
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN

class PE_Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, max_dim=256, logscale=True, max_value = 1000.0):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PE_Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        self.max_value = max_value
        self.max_dim = max_dim

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        
        x_ = x / self.max_value # yufan, normalize

        x_ = torch.Tensor([x_]).type_as(x)

        out = [x_]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x_)]

        # append 0s at the end, because if N_freqs is too big, the calculation will get Nan. e.g., sin ( 2 ^ 100 * 800 )
        out = torch.cat(out, -1)
        out = torch.nn.functional.pad(out, (0, self.max_dim - 1 - 2 * self.N_freqs), "constant", 0)

        return out


from geotransformer.modules.ops.transformation import apply_transform
@torch.no_grad()
def get_distance_and_overlap(
        ref_nodes,
        src_nodes,
        ref_knn_points,
        src_knn_points,
        transform,
        pos_radius,
        ref_masks=None,
        src_masks=None,
        ref_knn_masks=None,
        src_knn_masks=None
):
    r"""
    Generate ground truth node correspondences.
    Each node is composed of its k nearest points. A pair of points match if the distance between them is below
    `self.pos_radius`.

    :param ref_nodes: torch.Tensor (M, 3)
    :param src_nodes: torch.Tensor (N, 3)
    :param ref_knn_points: torch.Tensor (M, K, 3)
    :param src_knn_points: torch.Tensor (N, K, 3)
    :param transform: torch.Tensor (4, 4)
    :param pos_radius: float
    :param ref_masks: torch.BoolTensor (M,) (default: None)
    :param src_masks: torch.BoolTensor (N,) (default: None)
    :param ref_knn_masks: torch.BoolTensor (M, K) (default: None)
    :param src_knn_masks: torch.BoolTensor (N, K) (default: None)

    :return corr_indices: torch.LongTensor (num_corr, 2)
    :return corr_overlaps: torch.Tensor (num_corr,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    if ref_masks is not None and src_masks is not None:
        node_masks = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))
    else:
        node_masks = None

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_distances = torch.sqrt(((ref_knn_points - ref_nodes.unsqueeze(1)) ** 2).sum(-1))  # (M, K)
    if ref_knn_masks is not None:
        ref_knn_distances[~ref_knn_masks] = 0.
    ref_radius = ref_knn_distances.max(1)[0]  # (M,)
    src_knn_distances = torch.sqrt(((src_knn_points - src_nodes.unsqueeze(1)) ** 2).sum(-1))  # (N, K)
    if src_knn_masks is not None:
        src_knn_distances[~src_knn_masks] = 0.
    src_radius = src_knn_distances.max(1)[0]  # (N,)
    dist_map = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    masks = torch.gt(ref_radius.unsqueeze(1) + src_radius.unsqueeze(0) + pos_radius - dist_map, 0)  # (M, N)
    if node_masks is not None:
        masks = torch.logical_and(masks, node_masks)
    ref_indices, src_indices = torch.nonzero(masks, as_tuple=True)

    if ref_knn_masks is not None and src_knn_masks is not None:
        ref_knn_masks = ref_knn_masks[ref_indices]  # (B, K)
        src_knn_masks = src_knn_masks[src_indices]  # (B, K)
        node_knn_masks = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))
    else:
        node_knn_masks = None

    # compute overlaps
    ref_knn_points = ref_knn_points[ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[src_indices]  # (B, K, 3)
    dist_map = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    if node_knn_masks is not None:
        dist_map[~node_knn_masks] = 1e12

    # import pdb
    # pdb.set_trace()

    point_corr_map = torch.lt(dist_map, pos_radius ** 2)
    ref_overlap_counts = torch.count_nonzero(point_corr_map.sum(-1), dim=-1).float()
    src_overlap_counts = torch.count_nonzero(point_corr_map.sum(-2), dim=-1).float()
    if node_knn_masks is not None:
        ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()
        src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()
    else:
        ref_overlaps = ref_overlap_counts / ref_knn_points.shape[1]  # (B,)
        src_overlaps = src_overlap_counts / src_knn_points.shape[1]  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    masks = torch.gt(overlaps, 0)
    ref_corr_indices = ref_indices[masks]
    src_corr_indices = src_indices[masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[masks]

    return corr_indices, corr_overlaps

def get_soft_mask(
        ref_knn_points,
        src_knn_points,
        transform,
        pos_radius,
        ref_points_f,
        src_points_f,
        ref_knn_masks=None,
        src_knn_masks=None
):
    r"""
    Generate ground truth node correspondences.
    Each node is composed of its k nearest points. A pair of points match if the distance between them is below
    `self.pos_radius`.

    :param ref_knn_points: torch.Tensor (M, K, 3)
    :param src_knn_points: torch.Tensor (N, K, 3)
    :param transform: torch.Tensor (4, 4)
    :param pos_radius: float
    :param ref_knn_masks: torch.BoolTensor (M, K) (default: None)
    :param src_knn_masks: torch.BoolTensor (N, K) (default: None)
    """
    K = ref_knn_points.shape[1]

    ref_knn_points = ref_knn_points.view([-1, 3])
    ref_knn_masks = ref_knn_masks.view([-1])
    src_knn_points = src_knn_points.view([-1, 3])
    src_knn_masks = src_knn_masks.view([-1])
    src_points_f_proj = transform.float() @ torch.nn.functional.pad(src_points_f, (0,1), "constant", 1).T
    src_points_f_proj = src_points_f_proj.T[:, :3]
    ref_points_f_proj = torch.inverse(transform.float()) @ torch.nn.functional.pad(ref_points_f, (0,1), "constant", 1).T
    ref_points_f_proj = ref_points_f_proj.T[:, :3]


    ref_knn_masks = ref_knn_masks[:, None].repeat([1, src_points_f.shape[0]])
    src_knn_masks = src_knn_masks[:, None].repeat([1, ref_points_f.shape[0]])
    dist_map1 = torch.cdist(ref_knn_points, src_points_f_proj, p=2)  
    dist_map1[~ref_knn_masks] = 1e12
    dist_map2 = torch.cdist(src_knn_points, ref_points_f_proj, p=2)   
    dist_map2[~src_knn_masks] = 1e12
    dist_map1 = dist_map1.view([-1, K * src_points_f.shape[0]])
    dist_map2 = dist_map2.view([-1, K * ref_points_f.shape[0]])

    min_dis_ref, _ = torch.min(dist_map1, dim = 1)
    min_dis_src, _ = torch.min(dist_map2, dim = 1)

    mask_anchor_ref = torch.nn.functional.sigmoid((pos_radius -  min_dis_ref) / 0.001)
    mask_anchor_src = torch.nn.functional.sigmoid((pos_radius -  min_dis_src) / 0.001)

    return mask_anchor_ref, mask_anchor_src

class GeoTransformer(nn.Module):
    def __init__(self, cfg, parsed_args=None):
        super(GeoTransformer, self).__init__()
        self.parsed_args = parsed_args
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.using_geo_prior=cfg.train.using_geo_prior
        self.prior_min_points=cfg.train.prior_min_points
        self.using_2d_prior=cfg.train.using_2d_prior
        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
            parsed_args=parsed_args
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        
        self.variance = nn.Parameter(data=torch.Tensor([0]), requires_grad=True)      

    def forward(self, data_dict):
        output_dict = {} 

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        ref_point_to_node, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        src_point_to_node, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        estimated_transform = data_dict['estimated_transform'].detach()

        use_est = True
        reset = False
        corr_indices_f = get_correspondences(ref_points_f.detach().cpu().numpy(), src_points_f.detach().cpu().numpy(), estimated_transform.detach().cpu().numpy(), 0.0375)
        if corr_indices_f.shape[0] == 0:
            # import pdb; pdb.set_trace()
            mask_anchor_ref = None
            mask_anchor_src = None
            reset = True
        else:
            est_ref_idx_f =np.array( list(set(corr_indices_f[:,0].tolist())))
            est_src_idx_f =np.array (list(set(corr_indices_f[:,1].tolist())))
            mask_anchor_ref = None
            mask_anchor_src = None
    
        if reset or est_src_idx_f.shape[0]<self.prior_min_points or est_ref_idx_f.shape[0]<self.prior_min_points:
            if self.training:
                corr_indices_f = get_correspondences(ref_points_f.detach().cpu().numpy(), src_points_f.detach().cpu().numpy(), transform.detach().cpu().numpy(), 0.0375)
                est_ref_idx_f =np.array( list(set(corr_indices_f[:len(corr_indices_f)//2,0].tolist())))
                est_src_idx_f =np.array (list(set(corr_indices_f[:len(corr_indices_f)//2,1].tolist())))
            elif 'ref_corr_indices' in data_dict.keys():
                ref_corr_indices =data_dict['ref_corr_indices'][:self.prior_min_points*10]
                src_corr_indices =data_dict['src_corr_indices'][:self.prior_min_points*10]
                ref_prior_corr_points=ref_points[ref_corr_indices,:]
                src_prior_corr_points=src_points[src_corr_indices,:]
                k = 0
                src_dist_map_f = torch.sqrt(pairwise_distance(src_prior_corr_points.unsqueeze(0), src_points_f.unsqueeze(0)))  # (B, N, N)
                est_src_idx_f = src_dist_map_f.topk(k=k + 1, dim=2, largest=False)[1][0, :, 0]  # (B, N, k)
                ref_dist_map_f = torch.sqrt(pairwise_distance(ref_prior_corr_points.unsqueeze(0), ref_points_f.unsqueeze(0)))  # (B, N, N)
                est_ref_idx_f = ref_dist_map_f.topk(k=k + 1, dim=2, largest=False)[1][0, :, 0]  # (B, N, k)
            else:
                est_ref_idx_f =np.random.permutation(ref_points_f.shape[0])[: self.prior_min_points]
                est_src_idx_f =np.random.permutation(src_points_f.shape[0])[: self.prior_min_points]

        if use_est:
            # import pdb; pdb.set_trace()
            ref_overlapped_points_c_idx=ref_point_to_node[est_ref_idx_f]
            src_overlapped_points_c_idx=src_point_to_node[est_src_idx_f]
            src_overlapped_points_c_idx,ref_overlapped_points_c_idx=torch.unique(src_overlapped_points_c_idx),torch.unique(ref_overlapped_points_c_idx)
            src_points_c_idx=torch.arange(src_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
            ref_points_c_idx=torch.arange(ref_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
            src_no_overlapped_points_c_list=[i.item()  for i in src_points_c_idx if i not in src_overlapped_points_c_idx  ]
            src_no_overlapped_points_c_idx=torch.from_numpy(np.array(src_no_overlapped_points_c_list))
            ref_no_overlapped_points_c_list=[i.item()  for i in ref_points_c_idx if i not in ref_overlapped_points_c_idx  ]
            ref_no_overlapped_points_c_idx=torch.from_numpy(np.array(ref_no_overlapped_points_c_list))
         
        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c, ref_embeddings , src_embeddings = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            ref_overlapped_points_c_idx,src_overlapped_points_c_idx,
            ref_no_overlapped_points_c_idx,src_no_overlapped_points_c_idx,
            mask_anchor_ref,mask_anchor_src,
            time_step = data_dict["time_step"]
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 4. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 5. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 6 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 6.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 7. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 8. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config, parsed_args):
    model = GeoTransformer(config, parsed_args)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
