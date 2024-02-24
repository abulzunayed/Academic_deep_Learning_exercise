import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._model.zero_grad()
        output = self._model(x)
        loss = self._crit(output.float(), y.float())
        loss.backward()
        self._optim.step()
        return loss.item()



    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        prediction = self._model(x)
        loss = self._crit(prediction.float(),  y.float())
        return loss.item(), prediction


    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model = self._model.train()
        self.train_loss = 0.0
        for x,y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            else:
                x = x.to('cpu')
                y = y.to('cpu')
            self.train_loss += self.train_step(x,y)
        train_average_loss = self.train_loss/22
        return train_average_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model = self._model.eval()
        self.val_test_average_loss = 0.0
        with t.no_grad():
            for x,y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                else:
                    x = x.to('cpu')
                    y = y.to('cpu')
                average_loss, pred = self.val_test_step(x,y)
                self.val_test_average_loss += average_loss
        return self.val_test_average_loss


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        #TODO
        train_loss = []
        val_loss = []
        self.epoch_counter = 0
        while True:
            if self._cuda:
                self._model = self._model.cuda()
                self._crit = self._crit.cuda()
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            #TODO
            print("Epoch no: ", self.epoch_counter, "out of total ", epochs)
            if self.epoch_counter == epochs:
                break

            train_loss.append(self.train_epoch())
            val_loss.append(self.val_test())
            print("Training Loss: ", train_loss[self.epoch_counter], "Validation Loss: ", val_loss[self.epoch_counter])

            min_loss_idx = val_loss.index(min(val_loss))
            patience = self.epoch_counter - min_loss_idx
            if patience >= self._early_stopping_patience:
                break
            self.epoch_counter += 1

        return train_loss, val_loss
