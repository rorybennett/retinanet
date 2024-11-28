from os.path import join

import numpy as np
import torch


########################################################################################################################
# Custom Patience Class, only save best model, not latest.
########################################################################################################################
class EarlyStopping:
    def __init__(self, patience=100, delta=0.5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # If current score within this range, it is considered the same as previous score.
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimiser, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path)
        elif score < self.best_score + self.delta:
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimiser, save_path):
        """Saves model when validation loss decreases."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }, join(save_path, 'model_best.pth'))
        self.val_loss_min = val_loss
