import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from torch.jit import Final
from prism.model.diffusion.cross_attention import CrossAttention

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class VisualEncoder(nn.Module):
    """Visual Encoder
    """
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        
        super().__init__()
        block_channel = [64, 128, 256]
        self_attn_head = 8
        self_dim_feedforward = 512
        
        assert in_channels == 3, cprint(f"Visual Encoder only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        self.point_transfomer_encoder = nn.TransformerEncoderLayer(d_model=block_channel[2], nhead=self_attn_head,dim_feedforward= self_dim_feedforward,
                                                                   batch_first=True,dropout = 0.1)
        
        self.point_transformer = nn.TransformerEncoder(self.point_transfomer_encoder, num_layers=4)

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )            
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[VisualEncoder] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
  

    def category_max_pool(self, features, labels):
        invalid_mask = labels == -1 
        features = torch.where(invalid_mask.unsqueeze(-1).expand_as(features), torch.zeros_like(features), features)
        B, N, F = features.shape
        labels += 1
        C = labels.max().item() + 1  
        labels_expanded = labels.unsqueeze(-1).expand(B, N, F)
        if features.dtype == torch.float32 or features.dtype == torch.float64:
            output = torch.full((B, C, F), -float('inf'), device=features.device, dtype=features.dtype)
        else:
            output = torch.full((B, C, F), torch.iinfo(features.dtype).min, device=features.device, dtype=features.dtype)
        output = output.scatter_reduce(1, labels_expanded, features, reduce='amax', include_self=True)
        output = output.masked_fill(output == -float('inf'), 0)
        return output
    
    def forward(self, pn_feat,labels):
        pn_feat = self.mlp(pn_feat)
        pn_feat = self.category_max_pool(pn_feat,labels)  
        pn_feat = self.point_transformer(pn_feat)
        return pn_feat
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()




class PrismEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.pn_state_cross_dim = 256
        self.cross_attn_head = 8
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.point_label_key = 'point_label'
        self.n_output_channels = out_channel
     
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[PrismEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[PrismEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[PrismEncoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.extractor = VisualEncoder(**pointcloud_encoder_cfg)
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
       
        self.pn_state_corss_attention1 = CrossAttention(dim=self.pn_state_cross_dim,num_heads=self.cross_attn_head,
                                                         attn_drop=0.0,proj_drop=0.1,qk_norm=True,qkv_bias=False)
        self.pn_state_corss_attention2 = CrossAttention(dim=self.pn_state_cross_dim,num_heads=self.cross_attn_head,
                                                         attn_drop=0.0,proj_drop=0.0,qk_norm=True,qkv_bias=False)
        self.cross_attn_mlp = nn.Sequential(
                nn.Linear(self.pn_state_cross_dim, self.pn_state_cross_dim*2),
                nn.LayerNorm(self.pn_state_cross_dim*2),
                nn.ReLU(),
                nn.Linear(self.pn_state_cross_dim*2, self.pn_state_cross_dim),
                nn.LayerNorm(self.pn_state_cross_dim),
                nn.ReLU(),
            )
    
        self.state_mlp = nn.Sequential(
                nn.Linear(7, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, self.pn_state_cross_dim),
                nn.LayerNorm(self.pn_state_cross_dim),
                nn.ReLU(),
            )
    
        self.final_mlp = nn.Sequential(
                nn.Linear(self.pn_state_cross_dim, 192),
                nn.LayerNorm(192),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.LayerNorm(96),
                nn.ReLU(),
            )
        cprint(f"[PrismEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        B = points.shape[0]
        if self.point_label_key in observations:
            idx = 0
            point_labels = observations[self.point_label_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] 
            points = torch.concat([points, img_points], dim=1)
    
        pn_feat = self.extractor(points,point_labels)      
        state = observations[self.state_key]
        state = state.view(B, 2, 7)
        state_feat = self.state_mlp(state) 
        cross_feat = self.pn_state_corss_attention1(state_feat,pn_feat)
        cross_feat = self.cross_attn_mlp(cross_feat)
        cross_feat = self.pn_state_corss_attention2(cross_feat,pn_feat)
        final_feat = self.final_mlp(cross_feat)
        final_feat = final_feat.view(3,-1)
    
        return final_feat
      

    def output_shape(self):
        return self.n_output_channels