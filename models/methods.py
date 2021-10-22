import json
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

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

    optimizer (torch.optim.optimizer.Optimizer): Function which
        backpropagates the gradients of the loss with respect to each
        parameter.

    scheduler (torch.optim.lr_scheduler): Function to call when changing
        the learning rate. Default value None

    Methods:
        train_model (self, torch.utils.data.DataLoader: train,
                    torch.utils.data.DataLoader: valid):

        save_model (self, int: epoch, double: val_loss):

        validate_model (self, torch.utils.data.DataLoader: dataloader):

        predict (self, torch.utils.data.DataLoader: dataloader):

        update_logs (self, str: key, double: value):
    """
    def __init__(self,
                 model,
                 epochs=10,
                 learning_rate=0.01,
                 criterion=nn.L1Loss,
                 optimizer=optim.Adam,
                 scheduler=None,
                 **kwargs):
        self.logs = {"loss": [], "batch_loss": [], "val_loss": [],
                     "epoch": [], "lr": []}
        self.model = model.to(DEVICE)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion().to(DEVICE)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.learning_rate)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **kwargs)

    def train_model(self, train, valid=None):
        for i in range(self.epochs):
            self.update_logs("epoch", i)
            if self.scheduler is not None:
                self.update_logs("lr", self.scheduler.get_last_lr())
            else:
                self.update_logs("lr", self.learning_rate)
            # reset losses and gradients on each epoch start
            accum_loss = 0
            total_loss = 0
            self.optimizer.zero_grad()

            for j, (inputs, nxt) in enumerate(train):
                inputs = inputs.to(DEVICE)
                nxt = nxt.to(DEVICE)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, nxt)
                accum_loss += loss.item()
                loss.backward()

                if j % 5 == 0 and j != 0:  # every 50 batches
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.update_logs("batch_loss", accum_loss / 5)
                    print(f"Batch {j}:\t loss = {accum_loss / 5}")
                    total_loss += accum_loss
                    accum_loss = 0
                    self.show_predictions(outputs, nxt, epoch=i, batch=j)

            total_loss *= 1 / len(train)
            self.update_logs("loss", total_loss)
            self.save_model(f"epoch{i}")
            self.save_logs("outputs/results/training")

            print(f"Epoch \t {i} finished, model saved")
            if self.scheduler is not None:
                self.scheduler.step()
            # Validation
            if valid is not None:
                self.validate_model(valid)

        return self.logs

    def save_model(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        loss = self.logs['loss'][-1]
        torch.save(self.model.state_dict(),
                   f"models/weights/{name}_{loss}_{timestamp}.pth")

    def validate_model(self, dataloader):
        with torch.no_grad():
            validation_loss = 0
            for i, (inputs, nxt) in enumerate(dataloader):

                inputs = inputs.to(DEVICE)
                nxt = nxt.to(DEVICE)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, nxt)
                validation_loss += loss.item()

            validation_loss *= 1/len(dataloader)
            self.update_logs("val_loss", validation_loss)

    def predict(self, dataloader):
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                return self.model(inputs)

    def show_predictions(self, outputs, true_batch, num_samples=1,
                         epoch=0, batch=0):
        fig, ax = plt.subplots(num_samples,
                               2,
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})

        for i, image in enumerate(outputs):
            ax[0].imshow(image.detach().numpy()[0])
            ax[1].imshow(true_batch[0].detach().numpy()[0])

            if i == (num_samples - 1):
                break
        fig.suptitle("Model predicted outputs", fontsize=24)
        ax[0].set_title('Predicted', fontsize=20)
        ax[0].set(xlabel=f"Epoch: {epoch}    Batch: {batch}")
        ax[1].set_title('Ground Truth', fontsize=20)

        # fig.supxlabel("")
        # fig.supylabel("")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"outputs/figures/training/{timestamp}.png")
        # plt.show()
        plt.close()

    def update_logs(self, key, value):
        self.logs[key].append(value)
        print(f"{key} : \t{value} \n")

    def save_logs(self, results_path):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        with open(f"{results_path}/logs_{timestamp}.json", 'w',
                  encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=4)
