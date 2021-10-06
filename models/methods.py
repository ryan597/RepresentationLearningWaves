import datetime
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


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

        validate_model:

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
        self.model = model.to(DEVICE)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion.to(DEVICE)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.learning_rate)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, scheduler_args)

    def train_model(self, train, valid):
        history = {"loss": [], "val_loss": [], "epoch": [], "lr": []}

        criterion = nn.L1Loss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=epochs*3,
                                                        eta_min=1e-7)

        for i in range(self.epochs):
            # update the logs
            history["epoch"].append(i+1)
            history["lr"].append(scheduler.get_lr())
            # reset losses and gradients on each epoch start
            accum_loss = 0
            total_loss = 0
            optimizer.zero_grad()
            print(f"learning rate {scheduler.get_lr()}")

            for j, (inputs, nxt) in enumerate(train):
                inputs = inputs.to(DEVICE)
                nxt = nxt.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, nxt)
                accum_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                if j % 10 == 0 and verbose:
                    print(f"loss \t {accum_loss / 10}")
                    total_loss += accum_loss
                    accum_loss = 0

            optimizer.step()
            optimizer.zero_grad()
            total_loss *= 1 / len(train)
            history["loss"].append(total_loss)

            print(f"Epoch \t {i+1} finished")
            scheduler.step()
            # Validation
            self.validate_model()

        return self.logs

    def save_model(self, epoch, valloss):
        timestamp = datetime.datetime.today().replace(second=0, microsecond=0)
        torch.save(self.model.state_dict(),
                   f"models/{epoch}_{valloss}_{timestamp}.pth")

    def validate_model(self, valid):
        with torch.no_grad():
            validation_loss = 0
            for i, (inputs, nxt) in enumerate(valid):

                inputs = inputs.to(DEVICE)
                nxt = nxt.to(DEVICE)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, nxt)
                validation_loss += loss.item()

            validation_loss *= 1/len(valid)
            history["val_loss"].append(valid)
            print(f"validation loss : \t{valid_loss} \n")

    def predict(self, x):
        pass

    def update_logs(self, key, value):
        self.logs[key].append(value)

    def get_logs(self, key=None):
        if key is None:
            return self.logs
        else:
            return self.logs[key]
