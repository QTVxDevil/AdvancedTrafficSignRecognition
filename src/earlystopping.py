import torch
from src.config import EARLY_STOPPING_PARAMS

class EarlyStopping:
    def __init__(self, patience=None, delta=None, verbose=None):
        self.patience = patience or EARLY_STOPPING_PARAMS['patience']
        self.delta = delta or EARLY_STOPPING_PARAMS['delta']
        self.verbose = verbose if verbose is not None else EARLY_STOPPING_PARAMS['verbose']
        self.counter = 0
        self.best_score = None  
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, checkpoint_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss
