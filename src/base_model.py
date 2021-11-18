import torch
import torch.nn as nn
from config.config import TaskConfig


class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)


class CRNN(nn.Module):

    def __init__(self, config: TaskConfig, max_window_length: int = 7):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
                                  config.stride[0] + 1

        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.streaming = False
        self.max_window_length = max_window_length  # crnn output size
        self.T = (max_window_length - 1) * config.stride + config.kernel_size  # melspec size
        self.slide = config.stride
        self.spec_buffer = None
        self.crnn_buffer = None

    def stream_on(self):
        self.spec_buffer = torch.Tensor([]).to(self.U.weight.device)
        self.crnn_buffer = torch.Tensor([]).to(self.U.weight.device)
        self.streaming = True

    def stream_off(self):
        self.spec_buffer = None
        self.crnn_buffer = None
        self.streaming = False

    def forward(self, batch, hidden=None):
        if self.streaming:
            batch = torch.cat((self.spec_buffer, batch), 2)
        batch = batch.unsqueeze(dim=1)

        conv_output = self.conv(batch).transpose(-1, -2)
        gru_output, hidden = self.gru(conv_output, hidden)

        contex_vector = self.attention(gru_output)
        logits = self.classifier(contex_vector)
        return (logits, hidden) if self.streaming else logits
