from ..dataset import LogMelspec
from ..base import CRNN
from ..base import BaseTrainer
from torch.utils.data import DataLoader
from config.config import TaskConfig


class Trainer(BaseTrainer):
    def __init__(
            self,
            config: TaskConfig, model: CRNN,
            train_loader: DataLoader, val_loader: DataLoader,
            melspec_train: LogMelspec, melspec_val: LogMelspec
    ):
        super().__init__(config, model, train_loader, val_loader, melspec_train, melspec_val)
