from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from src.base_trainer import BaseTrainer
from config.config import TaskConfig
from src.base_model import CRNN
from torch.utils.data import DataLoader
from src.preprocessing import LogMelspec
from IPython.display import clear_output


class DistillTrainer(BaseTrainer):
    def __init__(
            self,
            config: TaskConfig, teacher: CRNN, model: CRNN,
            train_loader: DataLoader, val_loader: DataLoader,
            melspec_train: LogMelspec, melspec_val: LogMelspec
    ):
        super().__init__(config, model, train_loader, val_loader, melspec_train, melspec_val)
        self.teacher = teacher
        self.temperature = config.temperature

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def _train_epoch(self, epoch):
        self.model.train()
        self.teacher.train()
        for idx, (batch, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batch, labels = batch.to(self.device), labels.to(self.device)
            batch = self.melspec_train(batch)

            self.optimizer.zero_grad()

            teacher_logits = self.teacher(batch) / self.temperature
            soft_labels = F.softmax(teacher_logits, dim=-1)

            logits = self.model(batch)

            soft_loss = F.cross_entropy(logits / self.temperature, soft_labels)
            hard_loss = F.cross_entropy(logits, labels)
            loss = self.temperature ** 2 * soft_loss + hard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()

            if idx % self.log_every == 0:
                self.logger.log({
                    'loss': loss.item(),
                    'grad norm': self.get_grad_norm(),
                    'learning rate': self.scheduler.get_last_lr()[0]
                })

        self.scheduler.step()
