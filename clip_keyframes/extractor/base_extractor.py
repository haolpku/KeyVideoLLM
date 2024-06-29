import torch
import os
from abc import abstractmethod
from config.base_config import Config


class BaseExtractor:
    """
    Base class for all extractors
    """
    def __init__(self, model, loss, metrics, optimizer, config: Config, writer=None):
        self.config = config
        # setup GPU device if available, move model into configured device
        self.device = self._prepare_device()
        self.model = model.to(self.device)

        self.loss = loss.to(self.device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.writer = writer
        self.checkpoint_dir = config.model_path

        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch



    @abstractmethod
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Training logic for a step in an epoch
        :param epoch: Current epoch number
               step: Current step in epoch
               num_steps: Number of steps in epoch
        """
        raise NotImplementedError


    def validate(self):
        self._valid_epoch_step(0,0,0)

    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device


    def load_checkpoint(self, model_name):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']
        
        self.model.load_state_dict(state_dict)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded")

