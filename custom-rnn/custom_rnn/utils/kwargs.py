class Kwargs:
    
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        
    def __call__(self, **kwargs):
        
        for kwarg in kwargs:
            
            self.kwargs[kwarg] = kwargs[kwarg]
        
        return self.kwargs