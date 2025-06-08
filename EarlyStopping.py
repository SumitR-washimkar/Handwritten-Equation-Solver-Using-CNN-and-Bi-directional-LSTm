import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path='best_model.pth', verbose=True, mode='min'):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as improvement.
            save_path (str): Path to save the best model checkpoint.
            verbose (bool): Whether to print messages.
            mode (str): 'min' to stop when loss stops decreasing, 'max' to stop when metric stops increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.verbose = verbose
        self.mode = mode
        
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # Initialize comparison function based on mode
        if self.mode == 'min':
            self.is_improvement = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.is_improvement = lambda current, best: current > best + self.min_delta
            self.best_score = -float('inf')

    def __call__(self, current_score, model):
        """
        Call to check early stopping condition.
        
        Args:
            current_score (float): Current value of the monitored metric (e.g., validation loss or accuracy).
            model (nn.Module): Model to save if improvement.
        """
        if self.is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Improved metric to {current_score:.4f}. Model saved to {self.save_path}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement: {current_score:.4f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")