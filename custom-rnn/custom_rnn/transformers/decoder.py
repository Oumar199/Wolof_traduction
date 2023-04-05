
from torch.nn import MultiheadAttention
from typing import Union
from torch import nn
from torch.nn import functional as F
import torch

class TransformerDecoderO(nn.Module):
    
    def __init__(self, 
                value_size: Union[int, None] = None,
                key_size: Union[int, None] = None,
                model_size: int = 512,
                head_size: int = 8,
                ff_units: int = 100,
                drop_out: float = 0.1):
        
        super(TransformerDecoderO, self).__init__()
        
        # Initiate the drop out rate for later
        self.drop_out = drop_out
        
        # Let's add a first residual network
        self.residual_1 = nn.Identity()
        
        self.mask_multiheadattention = nn.MultiheadAttention(model_size, head_size, drop_out, kdim=key_size, vdim=value_size, batch_first=True)
        
        # let's add a first layer normalization to avoid vanishing gradient
        self.layer_normalization1 = nn.LayerNorm(model_size)
        
         # We have to add the second residual network
        self.residual_2 = nn.Identity()
        
        self.crossheadattention = nn.MultiheadAttention(model_size, head_size, drop_out, kdim=key_size, vdim=value_size, batch_first=True)
        
        # let's add a second layer normalization to avoid vanishing gradient
        self.layer_normalization2 = nn.LayerNorm(model_size)
        
        # We have to add, moreover, the feed forward neural sequence and another residual network
        self.residual_3 = nn.Identity()
        
        # add drop_out before each adding
        self.drop_out1 = nn.Dropout(drop_out)
        
        self.FF = nn.Sequential(
            nn.Linear(model_size, ff_units),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(ff_units, model_size)
        )
        
        # the third and last layer normalization
        self.layer_normalization3 = nn.LayerNorm(model_size)
    
    def forward(self, states, target, pad_mask: Union[torch.Tensor, None] = None, targ_mask: Union[torch.Tensor, None] = None):
        
        # change the type of the target
        target = target.type_as(next(self.parameters()))
        
        x_1 = self.residual_1(target)
        
        target = self.mask_multiheadattention(target, target, target, attn_mask = targ_mask)
        
        target = self.layer_normalization1(target[0].float() + x_1)
        
        x_2 = self.residual_2(target)
        
        cross_output = self.crossheadattention(target, states, states)
        
        # normalize the crossover output
        cross_output = self.layer_normalization2(cross_output[0].float() + x_2)
        
        x_3 = self.residual_3(cross_output)
        
        output = self.drop_out1(self.FF(cross_output)).type_as(x_3) + x_3
        
        # normalize the final output
        output = self.layer_normalization3(output.float())
        
        return output
