
class EarlyStopping(object):
    def __init__(self, mode='min', patience=10):
        self.mode = mode
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        if mode == 'min':
            self.is_better = lambda a, best: a < best
        if mode == 'max':
            self.is_better = lambda a, best: a > best

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        if self.is_better(value, self.best):
            self.num_bad_epochs = 0
            self.best = value
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
