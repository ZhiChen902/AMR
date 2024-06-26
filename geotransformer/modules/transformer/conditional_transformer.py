import torch.nn as nn

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
        
import torch
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


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1



class RPEConditionalTransformer(nn.Module):
    def __init__(
            self,
            blocks,
            d_model,
            num_heads,
            dropout=None,
            activation_fn='ReLU',
            return_attention_scores=False,
            parallel=False,
            parsed_args=None
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.parsed_args=parsed_args
        self.blocks = blocks
        layers = []

        self.use_position = hasattr(parsed_args, "position_encoding") and parsed_args.position_encoding

        if self.use_position:
            position_transfer = []
            position_transfer_self_oneway = []
            position_transfer_cross_oneway = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn,parsed_args=parsed_args))
                if self.use_position:
                    position_transfer.append(nn.Linear(256, 256))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn, parsed_args=parsed_args))
                if self.use_position:
                    position_transfer.append(nn.Linear(256, 256))
                    if self.parsed_args.pe_selfoneway:
                        position_transfer_self_oneway.append(nn.Linear(256, 256))
                    if self.parsed_args.pe_crossoneway:
                        position_transfer_cross_oneway.append(nn.Linear(256, 256))
        
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

        if self.use_position:
            self.position_encoding_layer = nn.Sequential(
                PE_Embedding(1, 128),
                nn.Linear(256, 256), 
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
            ) 
            self.position_transfer_layers = nn.ModuleList(position_transfer)
            self.position_transfer_self_oneway_layers = nn.ModuleList(position_transfer_self_oneway)
            self.position_transfer_cross_oneway_layers = nn.ModuleList(position_transfer_cross_oneway)

    def forward(self, feats0, feats1, embeddings0, embeddings1, ref_overlapped_points_c_idx,
                src_overlapped_points_c_idx,
                ref_no_overlapped_points_c_idx, src_no_overlapped_points_c_idx, mask_anchor_ref,
                mask_anchor_src, masks0=None, masks1=None, time_step=None):
        attention_scores = []
        if self.use_position:
            common_position_feature = self.position_encoding_layer(time_step)

        count = 0

        for i, block in enumerate(self.blocks):
            if self.use_position:
                position_feature_i = self.position_transfer_layers[i](common_position_feature)
                feats0 = feats0 + position_feature_i
                feats1 = feats1 + position_feature_i
            if block == 'self':
                ### step 1: self attention
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    # import pdb
                    # pdb.set_trace()

                    ### step 2: self one way attention
                    anchor_feats0 = feats0[0, ref_overlapped_points_c_idx, :].unsqueeze(0)
                    anchor_feats1 = feats1[0, src_overlapped_points_c_idx, :].unsqueeze(0)
                    if ref_no_overlapped_points_c_idx.shape[0] > 0 and src_no_overlapped_points_c_idx.shape[0] > 0:
                        ref_no_overlapped_points_c_idx = ref_no_overlapped_points_c_idx
                        nooverlap_feats0 = feats0[0, ref_no_overlapped_points_c_idx, :].unsqueeze(0)
                        nooverlap_feats1 = feats1[0, src_no_overlapped_points_c_idx, :].unsqueeze(0)

                        if self.use_position:
                            if self.parsed_args.pe_selfoneway:
                                position_feature_i = self.position_transfer_self_oneway_layers[count](common_position_feature)
                                feats0 = feats0 + position_feature_i
                                feats1 = feats1 + position_feature_i      
                        
                        feats0[0, ref_no_overlapped_points_c_idx, :], _ = self.layers[i](nooverlap_feats0,
                                                                                        anchor_feats0,
                                                                                        memory_masks=masks0)
                        feats1[0, src_no_overlapped_points_c_idx, :], _ = self.layers[i](nooverlap_feats1,
                                                                                        anchor_feats1,
                                                                                        memory_masks=masks1)       
                    ### step 3: cross attention
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)  ###up
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)  ###up

                    ### step 4: cross one way attention
                    anchor_feats0 = feats0[0, ref_overlapped_points_c_idx, :].unsqueeze(0)
                    anchor_feats1 = feats1[0, src_overlapped_points_c_idx, :].unsqueeze(0)
                    
                    if ref_no_overlapped_points_c_idx.shape[0] > 0 and src_no_overlapped_points_c_idx.shape[0] > 0:
                        ref_no_overlapped_points_c_idx = ref_no_overlapped_points_c_idx
                        nooverlap_feats0 = feats0[0, ref_no_overlapped_points_c_idx, :].unsqueeze(0)
                        nooverlap_feats1 = feats1[0, src_no_overlapped_points_c_idx, :].unsqueeze(0)
                                            
                        if self.parsed_args.pe_crossoneway:
                            if self.use_position:
                                position_feature_i = self.position_transfer_cross_oneway_layers[count](common_position_feature)
                                feats0 = feats0 + position_feature_i
                                feats1 = feats1 + position_feature_i

                        if self.parsed_args.one_way_cross_type == 'non_to_anchor':
                            feats0_copy = feats0.clone()
                            feats1_copy = feats1.clone()
                            feats0_copy[0, ref_no_overlapped_points_c_idx, :], _ = self.layers[i](nooverlap_feats0,
                                                                                        anchor_feats1,
                                                                                        memory_masks=masks1)
                            feats0 = feats0_copy 
                            feats1_copy[0, src_no_overlapped_points_c_idx, :], _ = self.layers[i](nooverlap_feats1,
                                                                                        anchor_feats0,
                                                                                        memory_masks=masks0)
                            feats1 = feats1_copy
                        elif self.parsed_args.one_way_cross_type == 'anchor_to_anchor':
                            feats0_copy = feats0.clone()
                            feats1_copy = feats1.clone()
                            feats0_copy[0, ref_overlapped_points_c_idx, :], _ = self.layers[i](anchor_feats0,
                                                                                        anchor_feats1,
                                                                                        memory_masks=masks1)
                            feats0 = feats0_copy 
                            feats1_copy[0, src_overlapped_points_c_idx, :], _ = self.layers[i](anchor_feats1,
                                                                                        anchor_feats0,
                                                                                        memory_masks=masks0)
                            feats1 = feats1_copy
                        elif self.parsed_args.one_way_cross_type == 'all_to_anchor':
                            feats0, _ = self.layers[i](feats0,
                                                        anchor_feats1,
                                                        memory_masks=masks1)
                            feats1, _ = self.layers[i](feats1,
                                                        anchor_feats0,
                                                        memory_masks=masks0)
                count += 1
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
            self,
            blocks,
            d_model,
            num_heads,
            num_embeddings,
            dropout=None,
            activation_fn='ReLU',
            return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
