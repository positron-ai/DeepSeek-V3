from typing import Union, Optional
import torch
import os

class ValueLogger:
    """Utility class for logging intermediate tensor values during model execution"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Clear/create the log file
        open(log_file, 'w').close()
        
    def log_value(self, label: str, value: torch.Tensor, token_idx: int, 
                  layer: Optional[int] = None, expert: Optional[int] = None):
        """Log a tensor value with appropriate formatting"""
        # Convert tensor to flat list of floats
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().flatten().tolist()
        
        # Format the values as space-separated strings
        value_str = ' '.join(f'{v:.6f}' for v in value)
        
        # Build the prefix parts
        prefix = f"{label} of token {token_idx}"
        if layer is not None:
            prefix += f" layer {layer}"
        if expert is not None:
            prefix += f" expert {expert}"
            
        # Right justify to 50 characters
        prefix = prefix.rjust(50)
            
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"{prefix}: {value_str}\n") 

    def log_value_for_sequence(self, label: str, values: torch.Tensor, layer: Optional[int] = None, expert: Optional[int] = None):
        """Log values for each position in a sequence.
        
        Args:
            label (str): Label for the logged value
            values (torch.Tensor): Tensor of shape (batch_size, seq_len, ...) containing values to log
            layer (Optional[int]): Layer number if applicable
            expert (Optional[int]): Expert number if applicable
        """
        seqlen = values.size(1)
        for i in range(seqlen):
            self.log_value(label, values[0, i], i, layer, expert) 
