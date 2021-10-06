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

        get_logs (self, str: key=None):

    """
    def __init__(self,
                 model,
                 epochs=10,
                 learning_rate=0.01,
                 criterion=nn.L1Loss,
                 optimizer=optim.Adam,
                 scheduler=None,
                 **kwargs):
        self.logs = {"loss": [], "val_loss": [], "epoch": [], "lr": []}
        self.model = model.to(DEVICE)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion.to(DEVICE)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.learning_rate)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **kwargs)

    def train_model(self, train, valid=None):
        for i in range(self.epochs):
            self.update_logs("epoch", i)
            if self.scheduler is not None:
                self.update_logs("lr", self.scheduler.get_lr())
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

                self.optimizer.step()
                self.optimizer.zero_grad()
                if j % 10 == 0:  # every 10 batches
                    print(f"loss \t {accum_loss / 10}")
                    total_loss += accum_loss
                    accum_loss = 0

            total_loss *= 1 / len(train)
            self.update_logs("loss", total_loss)

            print(f"Epoch \t {i} finished")
            if self.scheduler is not None:
                self.scheduler.step()
            # Validation
            if valid is not None:
                self.validate_model()

        return self.logs

    def save_model(self, epoch, val_loss):
        timestamp = datetime.datetime.today().replace(second=0, microsecond=0)
        torch.save(self.model.state_dict(),
                   f"models/weights/{epoch}_{val_loss}_{timestamp}.pth")

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

    def update_logs(self, key, value):
        self.logs[key].append(value)
        print(f"{key} : \t{value} \n")

    def get_logs(self, key=None):
        if key is None:
            return self.logs
        else:
            return self.logs[key]


"""
def show_validation_predictions(valid, model, savefile=None):
    with torch.no_grad():
        for i, (batch_pair, batch_nxt) in enumerate(valid):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,
                                                          figsize=(15, 15))
            for j, img_pair in enumerate(batch_pair):
                ax1.imshow(img_pair[0])
                ax1.axis('off')
                ax2.imshow(img_pair[1])
                ax2.axis('off')
                ax3.imshow(batch_nxt[0][0])
                ax3.axis('off')
                pr = model(batch_pair.to(DEVICE))[0][0].detach()\
                    .to('cpu').numpy()
                pm = (pr - np.min(pr))/(np.max(pr)-np.min(pr))
                ax4.imshow(pm)
                ax4.axis('off')
                bnxt = ((batch_nxt[0][0].numpy() -
                        np.min(batch_nxt[0][0].numpy())) /
                        (np.max(batch_nxt[0][0].numpy()) -
                        np.min(batch_nxt[0][0].numpy())))
                ax5.imshow(1 - np.abs(img_pair[1] - bnxt), cmap='gray')
                ax5.axis('off')
                if savefile is not None:
                    plt.savefig(f"{savefile}_{i}")
                fig.show()
                break
            if i == 4:
                break
"""
