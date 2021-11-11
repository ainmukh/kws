import torch
from src.dataset_utils import DatasetDownloader, TrainDataset
from src.augmentations import AugsCreation
from src.sampler import get_sampler, Collator
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from src.preprocessing import LogMelspec
from collections import defaultdict
from IPython.display import clear_output
from matplotlib import pyplot as plt
from src.trainer import train_epoch, validation
from src.model import CRNN, AttnMech, FullModel
from config import TaskConfig


dataset_downloader = DatasetDownloader(TaskConfig.keyword)
labeled_data, background_noises = dataset_downloader.generate_labeled_data()

labeled_data.sample(3)


indexes = torch.randperm(len(labeled_data))
train_indexes = indexes[:int(len(labeled_data) * 0.8)]
val_indexes = indexes[int(len(labeled_data) * 0.8):]


train_df = labeled_data.iloc[train_indexes].reset_index(drop=True)
val_df = labeled_data.iloc[val_indexes].reset_index(drop=True)


# Sample is a dict of utt, word and label
transform_tr = AugsCreation()
train_set = TrainDataset(df=train_df, kw=TaskConfig.keyword, transform=transform_tr)
val_set = TrainDataset(df=val_df, kw=TaskConfig.keyword)


train_sampler = get_sampler(train_set.df['label'].values)
val_sampler = get_sampler(val_set.df['label'].values)


# Here we are obliged to use shuffle=False because of our sampler with randomness inside.
train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                          shuffle=False, collate_fn=Collator(),
                          sampler=train_sampler)
#                           num_workers=2, pin_memory=True)

val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                        shuffle=False, collate_fn=Collator(),
                        sampler=val_sampler,
                        num_workers=2, pin_memory=True)


melspec_train = LogMelspec(is_train=True, config=TaskConfig)
melspec_val = LogMelspec(is_train=False, config=TaskConfig)


history = defaultdict(list)


CRNN_model = CRNN(TaskConfig)
attn_layer = AttnMech(TaskConfig)
full_model = FullModel(TaskConfig, CRNN_model, attn_layer)
full_model = full_model.to(TaskConfig.device)

print(full_model)

opt = torch.optim.Adam(full_model.parameters(),
                       lr=TaskConfig.learning_rate, weight_decay=TaskConfig.weight_decay)


for n in range(TaskConfig.num_epochs):

    train_epoch(full_model, opt, train_loader,
                melspec_train, TaskConfig.device)

    au_fa_fr = validation(full_model, val_loader,
                          melspec_val, TaskConfig.device)
    history['val_metric'].append(au_fa_fr)

    clear_output()
    plt.plot(history['val_metric'])
    plt.ylabel('Metric')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()

    print('END OF EPOCH', n)