import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import torchaudio


class DatasetDownloader:
    def __init__(self, key_word='sheila'):
        self.key_word = key_word
        self.datadir = "speech_commands"
        
        if os.path.isfile('speech_commands_v0.01.tar.gz'):
            print('Data is already downloaded.')
        else:
            print('Downloading data...')
            os.system('wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz')
            os.system('mkdir speech_commands && tar -C speech_commands -xvzf speech_commands_v0.01.tar.gz 1> log')
            print("Ready!")

        self.samples_by_target = {
            cls: [os.path.join(self.datadir, cls, name) for name in os.listdir("./{}/{}".format(self.datadir, cls))]
            for cls in os.listdir(self.datadir)
            if os.path.isdir(os.path.join(self.datadir, cls))
        }
        
        print('Classes:', ', '.join(sorted(self.samples_by_target.keys())[1:]))

    def generate_labeled_data(self):
        if os.path.isfile('labeled_data.csv'):
            print('Data is already labeled')
            labeled_data = pd.read_csv('labeled_data.csv')
            background_noises = pd.read_csv('background_noises.csv')
            return labeled_data, background_noises

        labeled_data = pd.DataFrame(columns=['name', 'word', 'label'])
        background_noises = pd.DataFrame(columns=['name'])

        print('Creating labeled dataframe:')

        for el in tqdm(self.samples_by_target.keys()):
            if el != '_background_noise_':
                for name in self.samples_by_target[el]:
                    word = name.split('/')[1]
                    if word == self.key_word:
                        label = 1
                    else:
                        label = 0
                    labeled_data = labeled_data.append({'name': name, 'word': word, 'label': label}, ignore_index=True)
            
            elif el == '_background_noise_':
                for name in self.samples_by_target[el]:
                    if 'README' not in name:
                        background_noises = background_noises.append(
                            {'name':name}, ignore_index=True
                        )

        labeled_data.to_csv('labeled_data.csv', index=False)
        background_noises.to_csv('background_noises.csv', index=False)

        return labeled_data, background_noises

        
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, root='', df=None, kw=None, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            df (pd.DataFrame): dataframe with annotations (filename, word and label).
            kw (string): keyword to spot.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.kw = kw
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        utt_name = self.root + self.df.loc[idx, 'name']
        utt = torchaudio.load(utt_name)[0].squeeze()
        word = self.df.loc[idx, 'word']
        label = self.df.loc[idx, 'label']
        
        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt, 'word': word, 'label': label}
        return sample
