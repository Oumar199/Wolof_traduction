
import torch
from torch import nn
from torch.nn import functional as F
from custom_rnn.transformers.add_position import PositionalEncoding
from custom_rnn.transformers.predict_size import SizePredict
from copy import deepcopy
from typing import Union

# new Exception for that transformer
class TargetException(Exception):
    
    def __init__(self, error):
        
        print(error)

class TransformerOMultiple(nn.Module):
    
    def __init__(self, 
                 input_size: int,
                 encoder,
                 decoder,
                 class_criterion = None,
                 size_criterion = nn.MSELoss(),
                 n_features: int = 100,
                 n_layers: int = 1,
                 n_poses_max: int = 500,
                 projection_type: str = "embedding",
                 pad_mask: bool = True,
                 output_size: Union[int, None] = None,
                 n_encoder: int = 1,
                 n_decoder: int = 1,
                 return_target_size: bool = True):
        
        super(TransformerOMultiple, self).__init__()
        
        # we can make multiple encoding and decoding passes
        self.n_encoder = n_encoder
        
        self.n_decoder = n_decoder
        
        # we can initiate the positional encoding model
        self.pe = PositionalEncoding(n_poses_max, encoder.multiheadattention.embed_dim)
        
        # we can initiate the return target size argument which specify if the model has to return the 
        # predicted target size
        self.return_target_size = return_target_size
        
        self.pad_mask = pad_mask
        
        if projection_type == "embedding":
            
            self.embedding_layer = nn.Embedding(input_size, encoder.multiheadattention.embed_dim)
        
        elif projection_type == "linear":
            
            self.embedding_layer = nn.Linear(input_size, encoder.multiheadattention.embed_dim)
        
        # initialize the first encoder and decoder
        self.encoder = encoder
        
        self.decoder = decoder
        
        # add the first encoder and decoder into a list of encoders and decoders and make, if necessary, multiple copies of them
        self.encoders = nn.ModuleList([encoder])
        self.decoders = nn.ModuleList([decoder])
        
        for _ in range(1,self.n_encoder):
            
            self.encoders.append(deepcopy(self.encoder))
        
        for _ in range(1,self.n_decoder):
            
            self.decoders.append(deepcopy(self.decoder))
        
        self.class_criterion = class_criterion
        
        self.size_criterion = size_criterion
        
        # let's initiate the mlp for predicting the target size
        self.size_prediction = SizePredict(
            encoder.multiheadattention.embed_dim,
            n_features=n_features,
            n_layers=n_layers,
            normalization=True, # we always use normalization
            drop_out=encoder.drop_out_
            )
        
        self.classifier = None
        
        if output_size != None:
            
            self.classifier = nn.Sequential(
                nn.Linear(decoder.mask_multiheadattention.embed_dim, output_size),
                nn.Softmax(dim = -1)
            )
        
    def forward(self, input_, target = None):
        
        # ---> Encoder prediction
        input_embed = self.embedding_layer(input_)
        
        # recuperate the last input (before position)
        last_input = input_embed[:, -1:]
       
        # add position to input_embedding
        input_embed = self.pe(input_embed)
        
        # recuperate the masks
        pad_mask1 = pad_mask2 = None
        
        if self.pad_mask and self.training:
            
            pad_mask1 = (input_ != 0).type_as(next(self.parameters()))
            
        # let's initialize the states in order to make, if necessary, multiple passes through encoders
        states = input_embed
        
        # nextly we can use the encoders and decoders initialized into the list of modules
        for encoder in self.encoders:
            
            states = encoder(states, pad_mask = pad_mask1)
        
        # ---> Decoder prediction
        # let's initiate the target if it is equal to None
        if target is None and not self.training:
            
            target = target = torch.ones((states.size(0), self.size_prediction(states[:, -1]).round().long()[0].item())).long().to(next(self.parameters()).device)
        
        elif target is None:
            
            raise TargetException("You must provide a target if the model is on training mode !")
        
        target_embed = self.embedding_layer(target)
            
        if self.pad_mask and self.training:
        
            pad_mask2 = (torch.cat((input_[:, -1:], target[:, :-1]), dim=1) != 0).type_as(next(self.parameters()))
        
        targ_mask = self.get_target_mask(target_embed.size(1)).type_as(next(self.parameters()))
        
        # let's concatenate the last input and the target shifted from one position to the right (new seq dim = target seq dim)
        target_embed = torch.cat((last_input, target_embed[:, :-1]), dim = 1)
        
        # add position to target embed
        target_embed = self.pe(target_embed)
        
        # let's initialize the outputs in order to make, if necessary, multiple passes through decoders
        outputs = target_embed 
        
        # let's predict the size of the target 
        if self.training:
            
            target_size = self.size_prediction(states).mean()
            
        # we pass all of the shifted target sequence to the decoder if training mode
        if self.training:
            
            for decoder in self.decoders:
                
                outputs = decoder(states, outputs, pad_mask = pad_mask2, targ_mask = targ_mask)
        
        else: ## This part was understand with the help of the professor Bousso.
            
            # if we are in evaluation mode we will not use the target but the outputs to make prediction and it is
            # sequentially done (see comments)
        
            # the first output is the last input
            outputs = last_input
            
            # for each target that we want to predict
            for t in range(target.size(1)):
                
                # recuperate the target mask of the current decoder input
                current_targ_mask = targ_mask[:t+1, :t+1] # all attentions between the elements before the last target
                
                # we do the same for the padding mask
                current_pad_mask = None
                
                if pad_mask2 != None:
                    
                    current_pad_mask = pad_mask2[:, :t+1]
                
                # let's initialize the output of the first decoder
                out = outputs
                
                # make new predictions
                for decoder in self.decoders:
                    
                    out = decoder(states, out, pad_mask = current_pad_mask, targ_mask = current_targ_mask) 
                
                # add the last new prediction to the decoder inputs
                outputs = torch.cat((outputs, out[:, -1:]), dim = 1) # the prediction of the last output is the last to add (!)
            
            # let's take only the predictions (the last input will not be taken)
            outputs = outputs[:, 1:]
        
        # ---> Loss Calculation
        # let us calculate the loss of the size prediction
        size_loss = 0
        if not self.size_criterion is None and self.training:
            
            size_loss = self.size_criterion(target_size, torch.full((target.size(0), 1), target.size(1)).type_as(next(self.parameters())))
        
        if not self.classifier is None:
            
            outputs = self.classifier(outputs)
            
            if not self.class_criterion is None:
                
                loss = 0
                
                for s in range(outputs.size(1)):
                
                    loss += self.class_criterion(outputs[:, s, :], target[:, s])

                if self.return_target_size and self.training:
                    
                    return loss + size_loss, outputs, target_size
                
                return loss + size_loss, outputs
        
        if self.return_target_size and self.training:
            
            return outputs, target_size
        
        return outputs
    
    def get_target_mask(self, attention_size: int):
        
        return (1 - torch.triu(torch.ones((attention_size, attention_size)), diagonal = 1)).bool()
