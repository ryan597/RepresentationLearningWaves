import json
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

#from focal import sigmoid_focal_loss
from torchvision.ops import sigmoid_focal_loss

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
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = sigmoid_focal_loss
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.learning_rate)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **kwargs)

    def train_model(self, train, valid=None):
        print("Begining training...", flush=True)
        for i in range(self.epochs):
            print("\n", flush=True)
            self.update_logs("epoch", i)
            """if self.scheduler is not None:
                    self.update_logs("lr", self.scheduler.get_last_lr())
                else:
                    self.update_logs("lr", self.learning_rate)
            """
            # reset losses and gradients on each epoch start
            accum_loss = torch.Tensor([0]).to("cuda:0")
            total_loss = torch.Tensor([0]).to("cuda:0")
            self.optimizer.zero_grad()

            for j, (inputs, nxt) in enumerate(train):
                inputs = inputs.to("cuda:0")
                nxt = nxt.to("cuda:0")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, nxt, reduction='sum')
                accum_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_logs("batch_loss", accum_loss.item()) ## divide by j size?
                total_loss += accum_loss
                accum_loss = torch.Tensor([0]).to("cuda:0")

                #    if j==50:  ## only once per epoch
                self.show_predictions(outputs, nxt, epoch=i, batch=j)
            total_loss *= 1 / (len(train)) 
            self.update_logs("loss", total_loss.item())
            self.save_model(f"epoch{i}")
            self.save_logs("outputs/results/training")
            print(f"Epoch \t {i} finished, model saved", flush=True)

            # Validation
            if valid is not None:
                valid_loss = self.validate_model(valid)

            if self.scheduler is not None:
                self.scheduler.step(total_loss)
        return self.logs

    def save_model(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        loss = self.logs['loss'][-1]
        torch.save(self.model.state_dict(),
                   f"models/weights/{name}_{loss}_{timestamp}.pth")

    def validate_model(self, dataloader):
        with torch.no_grad():
            valid_loss = torch.Tensor([0]).to("cuda:0")
            for i, (inputs, nxt) in enumerate(dataloader):

                inputs = inputs.to("cuda:0")
                nxt = nxt.to("cuda:0")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, nxt, reduction='sum')
                valid_loss += loss.item()

            self.show_predictions(outputs, inputs)
            valid_loss *= 1 / ( len(dataloader))
            self.update_logs("val_loss", valid_loss.item())
            return valid_loss

    def predict(self, dataloader):
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                inputs = inputs.to("cuda:0")
                return self.model(inputs)

    def show_predictions(self, outputs, true_batch, num_samples=3,
                         epoch=0, batch=0):
        fig, ax = plt.subplots(num_samples,
                               2,
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                               subplot_kw={'xticks': [], 'yticks': []})

        for i, image in enumerate(outputs):
            ax[i, 0].imshow(image.detach().cpu().numpy()[0], cmap='gray')
            ax[i, 1].imshow(true_batch[0].detach().cpu().numpy()[0], cmap='gray')

            if i == (num_samples - 1):
                break
        fig.suptitle("Frame prediction", fontsize=20)
        ax[0, 0].set_title('Predicted', fontsize=15)
        # ax[0].set(xlabel=f"Epoch: {epoch}    Batch: {batch}")
        ax[0, 1].set_title('Ground Truth', fontsize=15)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if epoch==0 and batch==0:
            location = "validation"
        else:
            location = "training"
        plt.savefig(f"outputs/figures/{location}/{timestamp}_{epoch}_{batch}.png")
        plt.show()
        plt.close()

    def update_logs(self, key, value):
        self.logs[key].append(value)
        print(f"{key} : \t{value}", flush=True)

    def save_logs(self, results_path):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        with open(f"{results_path}/logs_{timestamp}.json", 'w',
                  encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=4)

