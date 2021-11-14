import torch
from src.augmentations import AugsCreation
from src.sampler import get_sampler, Collator
from torch.utils.data import DataLoader
from src.preprocessing import LogMelspec
from collections import defaultdict
from IPython.display import clear_output
from matplotlib import pyplot as plt
from src.base_trainer import train_epoch, validation
from src.base_model import CRNN
from config.config import TaskConfig
from src.dataset import SpeechCommandDataset


def train():
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=TaskConfig.keyword
    )

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    # Sample is a dict of utt, word and label
    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set = SpeechCommandDataset(csv=val_df)

    # samplers
    train_sampler = get_sampler(train_set.csv['label'].values)
    val_sampler = get_sampler(val_set.csv['label'].values)

    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.
    train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            sampler=val_sampler,
                            num_workers=2, pin_memory=True)

    # melspecs
    melspec_train = LogMelspec(is_train=True, config=TaskConfig)
    melspec_val = LogMelspec(is_train=False, config=TaskConfig)

    history = defaultdict(list)

    config = TaskConfig()
    model = CRNN(config).to(config.device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # TRAIN
    for n in range(TaskConfig.num_epochs):
        train_epoch(model, opt, train_loader,
                    melspec_train, config.device)

        au_fa_fr = validation(model, val_loader,
                              melspec_val, config.device)
        history['val_metric'].append(au_fa_fr)
        if len(history['val_metric']) == 1 or au_fa_fr < history['val_metric'][-2]:
            torch.save(model.state_dict(), TaskConfig.save_to)

        clear_output()
        plt.plot(history['val_metric'])
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.grid()
        plt.show()

        print('END OF EPOCH {:2}; auc: {:1.5f}'.format(n, au_fa_fr))


if __name__ == "__main__":
    train()
