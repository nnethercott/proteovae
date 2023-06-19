import torch

# ADD DOCSTRINGS


class BaseTrainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.num_iters = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_loader, epochs, val_data=None):
        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(
                f'(beta: {self.model.beta:>.1f}, eta: {self.model.eta:>.1f}, gamma: {self.model.gamma:>.1f})')

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
        self.model.train()
        size = len(data_loader.dataset)

        for _, data in enumerate(data_loader):
            data = self._to_device(data)
            losses = self._train_iteration(data)

        # on epoch end
        return losses

    def _train_iteration(self, data_batch):
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
        X, Y = data
        X = X.to(self.device)
        Y = Y.to(self.device)

        return (X, Y)


class ScheduledTrainer(BaseTrainer):
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
        pass
