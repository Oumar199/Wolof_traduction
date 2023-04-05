
from typing import Union
from torch import nn
from torch.nn import functional as F
import torch

class TransformerEncoderO(nn.Module):
    
    def __init__(self,
                 value_size: Union[int, None] = None,
                 key_size: Union[int, None] = None,
                 model_size: int = 512,
                 head_size: int = 8,
                 ff_units: int = 40,
                 drop_out: float = 0.1):
                
        super(TransformerEncoderO, self).__init__()
        
        # Initiate the drop out rate for later
        self.drop_out_ = drop_out
        
        # Let's add a first residual network
        self.residual_1 = nn.Identity()
        
        self.multiheadattention = nn.MultiheadAttention(model_size, head_size, drop_out, kdim=key_size, vdim=value_size, batch_first=True)
        
        # let us add the first layer normalization 
        self.layer_normalization1 = nn.LayerNorm(model_size)
        
        # We have to add, moreover, the feed forward neural sequence and a second residual network
        self.residual_2 = nn.Identity()
        
        self.FF = nn.Sequential(
            nn.Linear(model_size, ff_units),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(ff_units, model_size)
        )
        
        # let's add the second layer normalization to avoid vanishing gradient 
        self.layer_normalization2 = nn.LayerNorm(model_size)
        
        # add a last drop_out
        self.drop_out = nn.Dropout(drop_out)
        
    def forward(self, x, pad_mask: Union[torch.Tensor, None] = None):
        
        # change type of input
        x = x.type_as(next(self.parameters()))
        
        # let us recuperate the input in order to make a simple residual block
        x_1 = self.residual_1(x)
        
        multi_head_output = self.layer_normalization1(self.multiheadattention(x, x, x)[0] + x_1)
        
        x_2 = self.residual_2(multi_head_output)
        
        # calculate the states (for the moment without the layer normalization)
        states = self.drop_out(self.FF(multi_head_output[0])) + x_2
        
        return self.layer_normalization2(states)
