import torch.nn as nn
import torch.optim as optim


class PyTorchModel():
    """
    Class for PyTorch models to easily train, validate and test the model.

    Arguments:
        model (nn.Module): An instance of nn.Module which has a defined
            forward() method
        epochs (int): The number of epochs to train for. Default value 10.
        learning_rate (double): The multiplication parameter to be passed when
            updating weights in backpropagation. Controls how large the weights
            updates can be. Default value 0.01
        criterion (torch.autograd.Function): Should be an instance of one of
            the pytorch loss functions or a callable function which returns a
            loss value. A custom function should use torch operations
            (in loss calculation and in the forward pass) for autograd.
            Otherwise a custom class which inherits from
            torch.autograd.Function and defines both forward and backward
            methods. Default value torch.nn.L1Loss()
        optimizer (torch.optim.optimizer.Optimizer):
        scheduler (): Function to call when changing the learning rate.
            Default value None

    Methods:
        train_model:
        save_model:
        evaluate_model:
        predict:
        update_logs:
        get_logs:

    """
    def __init__(self,
                 model,
                 epochs=10,
                 learning_rate=0.01,
                 criterion=nn.L1Loss,
                 optimizer=optim.Adam,
                 scheduler=None):
        self.logs = {"loss": [], "val_loss": [], "epoch": [], "lr": []}
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self):
        pass

    def save_model(self):
        pass

    def evaluate_model(self):
        pass

    def predict(self, x):
        pass

    def update_logs(self):
        pass

    def get_logs(self):
        pass
