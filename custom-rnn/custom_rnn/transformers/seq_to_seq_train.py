
from torch import nn
from torch.nn import functional as F
from custom_rnn.utils.kwargs import Kwargs
from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
import shutil
import torch
import os

class CheckpointsError(Exception):
    
    def __init__(self, error = None):
        
        if error is None:
            
            print("You must delete the existing checkpoints' file with the same name as you specified and retry again!")

        else:
            
            print(error)

class SequenceToSequenceTrain:
    
    def __init__(self, dataset, model, criterion = nn.CrossEntropyLoss(), optimizer = torch.optim.Adam, version: int = 0):
        
        assert not version is None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        
        # recuperate the version
        self.version = version
        
        self.dataset = dataset
        
        self.model = model
        
        self.criterion = criterion
        
        self.optimizer = optimizer
        
        self.model_kwargs = Kwargs(criterion = nn.CrossEntropyLoss())
        
        self.optimizer_kwargs = Kwargs(lr = 0.1)
        
        self.loader_kwargs = Kwargs(shuffle = True)
        
    def compile(self, batch_size:int = 1,
                model_kwargs: dict = {},
                optimizer_kwargs: dict = {},
                loader_kwargs: dict = {}, 
                clipping_value: Union[float, None] = None,
                checkpoint_directory: str = "custom_rnn/checkpoints/",
                checkpoint_file_name: str = "checkpoints"):
        
        self.model_kwargs_ = self.model_kwargs(**model_kwargs)
        
        self.optimizer_kwargs_ = self.optimizer_kwargs(**optimizer_kwargs)
        
        self.loader_kwargs_ = self.loader_kwargs(**loader_kwargs)
        
        self._model = self.model(**self.model_kwargs_).to(self.device)
        
        self._optimizer = self.optimizer(self._model.parameters(), **self.optimizer_kwargs_)
        
        self.loader = DataLoader(self.dataset, batch_size=batch_size, **self.loader_kwargs_)
        
        self.clipping_value = clipping_value
        
        self.directory = checkpoint_directory
        
        self.file_name = checkpoint_file_name
        
        self.current_epoch = None
        
    def train(self, epochs: int = 40, log_steps: int = 40, autosave: bool = True):
        
        loss = 0
        
        global_steps = 0
        
        first_epoch = 0 if self.current_epoch is None else self.current_epoch
        
        for epoch in tqdm(range(first_epoch, first_epoch + epochs)):
            
            for data, target in self.loader:
                
                data = data.long().to(self.device)
                
                target = target.long().to(self.device)
                
                # try:
                 
                loss, outputs = self._model(data, target)
                
                # except Exception as e:
                    
                #     print(e)
                
                loss.backward()
                
                # if clip value is specified then clip the gradients between - clip value and clip value
                if self.clipping_value:
                    nn.utils.clip_grad_value_(self._model.parameters(), clip_value=self.clipping_value)
                
                self._optimizer.step()
                
                self._optimizer.zero_grad()

                if global_steps % log_steps == 0:
                    
                    predictions = torch.argmax(outputs.data, dim = 2)
                    
                    print(f"Predicted: {self.dataset.traduce(predictions[0].tolist())}")
                    
                    print(f"True: {self.dataset.traduce(target[0].tolist())}")
                
                global_steps += 1
            
            self.current_epoch = epoch
            
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
    
    def save(self, directory: str = "custom_rnn/checkpoints/", file_name: str = "checkpoints", version: Union[int, None] = None, overwrite: bool = False):
        
        # verification of the validity of the directory path
        assert os.path.isdir(directory)
        
        if not os.path.exists(directory):
            
            raise OSError(f"The specified directory {directory} does not exist !")
        
        # let's recuperate the version
        version = version if not version is None else self.version 
        
        # let's create the path
        path = os.path.join(directory, f"version_{version}")
        
        # let's verify if the directory for the version does not already exist 
        if os.path.exists(path):
            
            print("The directory for that version already exist. The checkpoints will be saved in it.")
        
        else:
            
            os.makedirs(path)
            
        # let's create the checkpoints' file path
        file_path = os.path.join(path, f"{file_name}.pth")
        
        # let's verify if the file does not already exist 
        option = 'saved'
        
        if os.path.exists(file_path):
            
            # if the overwrite argument is to True then the existing file will be replaced
            if overwrite:
                
                os.remove(file_path)

                option = 'overwrited'
                
            else:
                
                raise CheckpointsError(f"A file with the name {file_name} already exists at the same directory! To overwrite that file change the overwrite argument to True!")
        
        # recuperate the ckeckpoints
        checkpoints = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'current_epoch': self.current_epoch,
        }
        
        # save the checkpoints
        torch.save(checkpoints, file_path)
        
        print(f"The file was successfully {option}!")
    
    def load(self, directory: str = "custom_rnn/checkpoints/", file_name: str = "checkpoints", version: Union[int, None] = None):
        
        # let's recuperate the version
        version = version if not version is None else self.version 
        
        # let's get the checkpoints' file path
        file_path = os.path.join(directory, os.path.join(f"version_{version}", f"{file_name}.pth"))
        
        # let us verify if the path exist
        if not os.path.exists(file_path):
            
            raise OSError(f"The file specified at the path {file_path} does not exist!")
        
        if os.path.isfile(file_path):
            
            # recuperate the ckeckpoints
            checkpoints = torch.load(file_path)
            
            # load the state dicts and other parameters
            self._model.load_state_dict(checkpoints['model_state_dict'])
            
            self._optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            
            self.current_epoch = checkpoints['current_epoch']

            print("The model was successfully loaded!")
        
        else:
            
            raise OSError(f"The specified checkpoints' path is not file!")
