from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from kws.metric.au_fa_fr import get_au_fa_fr
from config.config import TaskConfig
from kws.base.base_model import CRNN
from kws.dataset.preprocessing import LogMelspec
from IPython.display import clear_output


class BaseTrainer:
    def __init__(
            self,
            config: TaskConfig, model: CRNN,
            train_loader, val_loader,
            melspec_train: LogMelspec, melspec_val: LogMelspec
    ):
        self.config = config
        self.device = config.device
        self.logger = wandb

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.melspec_train = melspec_train
        self.melspec_val = melspec_val

        self.epochs = config.num_epochs
        self.log_every = config.log_every
        self.save_to = self.config.save_to
        self.history = list()

    def train(self) -> list:
        self.logger.init(project='KWS', config=self.config.__dict__)

        prev_auc = float('inf')
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            auc = self._valid_epoch()

            if auc < prev_auc:
                torch.save(self.model.state_dict(), self.save_to)
                prev_auc = auc

            clear_output()
            self.history.append(auc)
            print('END OF EPOCH {:2}; auc: {:1.6f}'.format(epoch + 1, auc))
        return self.history

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, (batch, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batch, labels = batch.to(self.device), labels.to(self.device)
            batch = self.melspec_train(batch)

            self.optimizer.zero_grad()

            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()

            if idx % self.log_every == 0:
                self.logger.log({
                    'loss': loss.item(),
                    'grad norm': self.get_grad_norm(),
                    'learning rate': self.optimizer.param_groups[0]['lr']
                })

    @torch.no_grad()
    def _valid_epoch(self) -> float:
        self.model.eval()
        probabilities = []
        total_labels = []
        losses = []
        for idx, (batch, labels) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            batch, labels = batch.to(self.device), labels.to(self.device)
            batch = self.melspec_val(batch)

            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1)
            loss = self.criterion(logits, labels)

            probabilities.append(probs[:, 1].cpu())
            total_labels.append(labels.cpu())
            losses.append(loss.item())

        auc_fa_fr = get_au_fa_fr(torch.cat(probabilities, dim=0).cpu(), total_labels)
        self.logger.log({
            'AUC_FA_FR': auc_fa_fr,
            'val. loss': sum(losses) / len(losses)
        })
        return auc_fa_fr

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _save_checkpoint(self, epoch, save_best: bool = False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        filename = self.save_to + "/checkpoint-epoch{}.pth".format(epoch)
        if not save_best:
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = self.save_to + "/model_best.pth"
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
