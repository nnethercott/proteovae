import torch
from ..models.utils import ModelOutput
from torch.optim import lr_scheduler


class BaseTrainer():
    """Trainer object handling optimzer steps, backpropogation, and device assertions for batch data

    Args:
        model (~torch.nn.Module): the object to be trained 
        optimizer (~torch.optim.Optimizer): an optimzer object handling gradient updates over training
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer
                 ):

        self.model = model
        self.optimizer = optimizer
        self.num_iters = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              epochs: int,
              val_data=None):
        r"""
        Training loop for models derived from :class:`~proteovae.models.models.GuidedVAE`

        Parameters:
            train_loader (~torch.utils.data.DataLoader): training data loader 
            epochs (int): number of passes over `train_loader`
            val_data: Optional tuple of torch.Tensors to assess the performance of model while training
        """
        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            
            #print(f'(beta: {self.model.beta:>.1f}, eta: {self.model.eta:>.1f}, gamma: {self.model.gamma:>.1f})')

            # Training
            losses = self._train_epoch(train_loader)
            print(losses)

            # Validation
            if val_data:
                vals = self._val_epoch(val_data)
                print(vals)

            print('\n')

        print(f'Done!')

    def _train_epoch(self, data_loader):
        """Single epoch train protocol 

        Parameters:
            data_loader (~torch.utils.data.DataLoader): training loader 
        Returns:
            losses (ModelOutput): last computed losses in the epoch 
        """

        self.model.train()
        size = len(data_loader.dataset)

        for _, data in enumerate(data_loader):
            data = self._to_device(data)
            losses = self._train_iteration(data)

        # on epoch end
        return losses

    def _train_iteration(self, data_batch):
        """Single train iteration 

        Args:
            data_batch (~torch.Tensor): batch from next(iter(train_loader))

        Returns:
            losses (ModelOutput): loss computed over the batch 
         """
        # update term weightings in elbo
        self.model._elbo_scheduler_update(self.num_iters)

        losses = self.model.loss_function(data_batch)
        loss = losses['loss']

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increment internal state
        self.num_iters += 1

        return losses

    def _val_epoch(self, data):
        self.model.eval()
        data = self._to_device(data)

        vals = self.model.val_function(data)
        return vals

    def _to_device(self, data):
        r"""
        Handles device side assertions for mapping the batch data to the `Trainer` device. 

        Args:
            data (~torch.Tensor, ~torch.Tensor): data and labels 
        Returns:
            data (~torch.Tensor, ~torch.Tensor): data on trainer device 
        """
        X, Y = data
        X = X.to(self.device)
        Y = Y.to(self.device)

        return (X, Y)


class ScheduledTrainer(BaseTrainer):
    r"""Trainer inheriting from :class:`BaseTrainer`.  Adds the functionality of adaptively scheduling learning rate

    Args:
        model (~torch.nn.Module): the object to be trained 
        optimizer (~torch.optim.Optimizer): an optimzer object handling gradient updates over training
        scheduler (~torch.optim.lr_scheduler.LRScheduler): an lr_scheduler object

    """

    def __init__(self, model, optimizer, scheduler):
        super(ScheduledTrainer, self).__init__(model, optimizer)
        self.scheduler = scheduler

        # self.loss_hist=[]

    def _train_epoch(self, data_loader):
        print(f'lr={self.scheduler.get_last_lr()[0]:.2e}')

        self.model.train()
        size = len(data_loader.dataset)

        for _, data in enumerate(data_loader):
            data = self._to_device(data)
            losses = self._train_iteration(data)
            # self.loss_hist.append({k:v.cpu().detach().numpy() for k,v in losses.items()})

        self.scheduler.step()

        # on epoch end; logging, etc
        self._on_epoch_end(losses)

        return losses

    def _on_epoch_end(self, metrics):
        """
        Empty method overwritten externally to provide a hook called on epoch end. Ex, print statements, saving intermediate results 
        """
        pass
