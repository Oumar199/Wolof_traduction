
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from custom_rnn.transformers.seq_to_seq_train import SequenceToSequenceTrain
from custom_rnn.utils.kwargs import Kwargs
from typing import Union
from tqdm import tqdm

class SequenceToSequenceTrainO(SequenceToSequenceTrain, SummaryWriter):
    
    def __init__(self, dataset, model, class_criterion = nn.CrossEntropyLoss(), size_criterion = nn.MSELoss(), optimizer = torch.optim.Adam, version: int = 0, log_dir: str = "transformer_logs"):
        
        SequenceToSequenceTrain.__init__(self, dataset, model, optimizer = optimizer, version = version)
        
        SummaryWriter.__init__(self, log_dir)
        
        self.logger_dir = os.path.join(log_dir, f"version_{version}")
        
        self.model_kwargs = Kwargs(class_criterion = class_criterion, size_criterion = size_criterion)
        
        self.optimizer_kwargs = Kwargs()
        
        self.loader_kwargs = Kwargs(shuffle = True)
    
    def train(self, epochs: int = 40, log_steps: int = 40, autosave: bool = True):
        
        loss_sum = 0
        
        global_steps = 0
        
        first_epoch = 0 if self.current_epoch is None else self.current_epoch
        
        for epoch in tqdm(range(first_epoch, first_epoch + epochs)):
            
            for data, target in self.loader:
                
                self._model.train()
                
                data = data.long().to(self.device)
                
                target = target.long().to(self.device)
                
                # try:
                target_size = None
                
                if hasattr(self._model, "return_target_size") and self._model.return_target_size:
                    
                    loss, outputs, target_size = self._model(data, target)
                
                else:
                    
                    loss, outputs = self._model(data, target)
                
                # except Exception as e:
                    
                #     print(e)
                
                loss_sum += loss.item()
                
                loss.backward()
                
                # if clip value is specified then clip the gradients between - clip value and clip value
                if self.clipping_value:
                    nn.utils.clip_grad_value_(self._model.parameters(), clip_value=self.clipping_value)
                
                self._optimizer.step()
                
                self._optimizer.zero_grad()

                if (global_steps + 1) % log_steps == 0:
                    
                    predictions = torch.argmax(outputs.data, dim = 2)
                    
                    print(f"Predicted: {self.dataset.traduce(predictions[0].tolist())}")
                    
                    print(f"True: {self.dataset.traduce(target[0].tolist())}")
                    
                    if not target_size is None:
                        
                        print(f"Predicted target size: {target_size.round()}")
                        
                        print(f"True target size: {predictions.size(1)}")
                
                global_steps += 1

            # let us recuperate the current epoch
            self.current_epoch = epoch
            
            # let's show the loss evolution at each epoch end
            with SummaryWriter(self.logger_dir, comment="adding metrics") as writer:
                
                writer.add_scalar('loss', loss_sum / len(self.loader), global_step=epoch+1)
            
            loss_sum = 0
            
            # save automatically the checkpoints
            if autosave:
                
                self.save(self.directory, self.file_name, overwrite=True)
    
    def predict(self, first_half_name: Union[str, list], target_length: Union[list, None] = None):
        """Predicts the second half of a name given the first half 

        Args:
            first_half_name (Union[str, list]): The first half of a name or a list of it
            target_length (Union[list, None]): The number of elements in the target sequence. Defaults to None.
        """
        
        # if a string is given we make a list with a unique element
        if type(first_half_name) is str: first_half_name = [first_half_name]
        
        if not target_length is None:
            
            assert type(target_length) is list and len(target_length) == len(first_half_name)
        
        # for each element we make a new prediction after transforming it to positions
        predictions = [] # initialize the predictions
        
        with torch.no_grad():
            
            for i, half_name in enumerate(first_half_name):
                
                positions = self.dataset.encode(half_name)
                
                # transform the positions to a tensor and add on it the device and transform to long type
                positions = torch.tensor(positions).long().to(self.device).unsqueeze(0)
                
                # let's create a fake target tensor
                if target_length is None:
                    
                    if hasattr(self._model, "size_prediction"):
                        
                        target = None
                    
                    else:
                    
                        target = torch.ones(positions.size()).long().to(self.device)
                
                else:
                    
                    target = torch.ones((1, target_length[i])).long().to(self.device)
                
                # let's make the model to evaluation mode
                self._model.eval()
                
                # let's predict the second half
                outputs = self._model(positions, target)
                
                if type(outputs) is tuple:
                    
                    outputs = outputs[1]
                
                prediction = torch.argmax(outputs.data, dim = -1)
                
                # let's decode the positions
                prediction_ = self.dataset.traduce(prediction[0])
                
                # let's add the new prediction to the list of predictions
                predictions.append(prediction_)
        
        # return the predictions
        return predictions
