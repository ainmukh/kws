from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
from src.metrics import get_au_fa_fr, count_FA_FR
from src.base_trainer import validation
from collections import defaultdict
from IPython.display import clear_output
from matplotlib import pyplot as plt


def train_epoch(
        teacher, model, opt, loader, log_melspec, device,
        temperature: int = 1, alpha: float = 1, beta: float = 1
):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        teacher_logits = teacher(batch) / temperature
        logits = model(batch)
        # we need probabilities so we use softmax & CE separately
        probs = F.softmax(logits, dim=-1)
        soft_labels = F.softmax(teacher_logits, dim=-1)

        loss_1 = F.cross_entropy(logits / temperature, soft_labels)
        loss_2 = F.cross_entropy(logits, labels)
        loss = alpha * temperature**2 * loss_1 + beta * loss_2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc


def train(
        teacher, model, model_name, opt, config,
        train_loader, val_loader,
        melspec_train, melspec_val,
        history,
        alpha: float = 0.5,
        beta: float = 0.5
):
    # history = defaultdict(list)
    for n in range(config.num_epochs):
        train_epoch(teacher, model, opt, train_loader,
                    melspec_train, config.device, config.temperature)

        au_fa_fr = validation(model, val_loader,
                              melspec_val, config.device)
        history[model_name].append(au_fa_fr)
        if len(history[model_name]) == 1 or au_fa_fr < history[model_name][-2]:
            torch.save(model.state_dict(), config.save_to)

        clear_output()
        plt.plot(history[model_name])
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.grid()
        plt.show()

        print('END OF EPOCH {:2}; auc: {:1.5f}'.format(n, au_fa_fr))

    return history
