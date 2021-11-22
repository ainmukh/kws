from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ..base import BaseTrainer
from ..base import CRNN
from ..dataset import LogMelspec
from ..metric import get_au_fa_fr
from config.config import TaskConfig
from IPython.display import clear_output
import logging


class PruneTrainer(BaseTrainer):
    def __init__(
            self,
            config: TaskConfig, model: CRNN,
            train_loader, val_loader,
            melspec_train: LogMelspec, melspec_val: LogMelspec
    ):
        super(PruneTrainer, self).__init__(
            config, model, train_loader, val_loader, melspec_train, melspec_val
        )
        self.prune_iter = config.prune_iter

    def _prune(self):
        aucs = []
        torch.save(self.model.state_dict(), 'saved/model_to_prune.pth')
        out_channels = self.model.conv[0].out_channels
        self.config.cnn_out_channels = out_channels
        for i in tqdm(range(self.model.conv[0].out_channels)):
            self._prune_conv(filter_idx=i)
            auc = self._get_rank()
            aucs.append(auc)

            self.model = CRNN(self.config).to(self.device)
            self.model.load_state_dict(
                torch.load('saved/model_to_prune.pth', map_location=self.device)
            )
        idx = np.argmin(aucs)
        self._prune_conv(idx)

    def _prune_conv(self, filter_idx):
        conv = self.model.conv[0]
        new_conv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels - 1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None)
        )

        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()

        new_weights[:filter_idx, :, :, :] = old_weights[:filter_idx, :, :, :]
        new_weights[filter_idx:, :, :, :] = old_weights[filter_idx + 1:, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weights).to(self.device)

        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_idx] = bias_numpy[:filter_idx]
        bias[filter_idx:] = bias_numpy[filter_idx + 1:]
        new_conv.bias.data = torch.from_numpy(bias).to(self.device)

        self.model.conv[0] = new_conv

        old_gru = self.model.gru.weight_ih_l0.data.cpu().detach().numpy()
        new_gru = np.zeros(shape=(old_gru.shape[0], old_gru.shape[1] - 18), dtype=np.float32)
        new_gru[:, :filter_idx * 18] = old_gru[:, :filter_idx * 18]
        new_gru[:, filter_idx * 18:] = old_gru[:, (filter_idx + 1) * 18:]
        new_gru = torch.nn.Parameter(torch.from_numpy(new_gru))

        self.model.gru.weight_ih_l0 = new_gru
        self.model.gru.input_size -= 18
        self.model = self.model.to(self.device)

    def prune_train(self):
        logging.basicConfig(level=logging.INFO)
        self.logger.init(project='KWS', config=self.config.__dict__)

        for iter in range(self.prune_iter):
            unprunned_auc = self._get_rank()
            self._prune()
            auc = float('inf')
            for epoch in range(10):
                self._train_epoch(epoch)
                auc = self._valid_epoch()
                if auc <= unprunned_auc:
                    torch.save(self.model.state_dict(), self.save_to)
                    print('END OF EPOCH {:2}; auc: {:1.6f}; prunned: {:2}'.format(epoch + 1, auc, iter + 1))
                    break
                clear_output()
                print('END OF EPOCH {:2}; auc: {:1.6f}'.format(epoch + 1, auc))
            if auc > unprunned_auc:
                logging.info(f"pruned {iter} filters")
                break

    def _get_rank(self) -> float:
        self.model.eval()
        probabilities = []
        total_labels = []
        losses = []
        for idx, (batch, labels) in enumerate(self.val_loader):
            batch, labels = batch.to(self.device), labels.to(self.device)
            batch = self.melspec_val(batch)

            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1)
            loss = self.criterion(logits, labels)

            probabilities.append(probs[:, 1].cpu())
            total_labels.append(labels.cpu())
            losses.append(loss.item())

        auc_fa_fr = get_au_fa_fr(torch.cat(probabilities, dim=0).cpu(), total_labels)
        return auc_fa_fr
