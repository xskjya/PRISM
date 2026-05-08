import numpy as np

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, warmup_strategy='linear'):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_strategy = warmup_strategy
        self.current_epoch = 0
        self.warmup_finished = False

        self._set_lr(0)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _linear_warmup(self, epoch):
        return self.base_lr * (epoch + 1) / self.warmup_epochs

    def _exponential_warmup(self, epoch):
        return self.base_lr * (2 ** (epoch + 1) - 1) / (2 ** self.warmup_epochs - 1)

    def _cosine_warmup(self, epoch):
        return self.base_lr * (1 - np.cos(np.pi * (epoch + 1) / self.warmup_epochs)) / 2

    def step(self, epoch=None):
        if epoch is None:
            self.current_epoch += 1
            epoch = self.current_epoch
        else:
            self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            if self.warmup_strategy == 'linear':
                lr = self._linear_warmup(epoch)
            elif self.warmup_strategy == 'exponential':
                lr = self._exponential_warmup(epoch)
            elif self.warmup_strategy == 'cosine':
                lr = self._cosine_warmup(epoch)
            else:
                raise ValueError(f"Unknown warmup strategy: {self.warmup_strategy}")

            self._set_lr(lr)
            return lr
        else:
            self.warmup_finished = True
            return None

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'warmup_finished': self.warmup_finished,
            'warmup_epochs': self.warmup_epochs,
            'base_lr': self.base_lr,
            'warmup_strategy': self.warmup_strategy
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.warmup_finished = state_dict['warmup_finished']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lr = state_dict['base_lr']
        self.warmup_strategy = state_dict['warmup_strategy']