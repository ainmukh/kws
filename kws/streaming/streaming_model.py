from kws.base import CRNN
import torch
from config.config import TaskConfig


class CRNNStreaming(CRNN):
    def __init__(self, config: TaskConfig, max_window_length: int = 7):
        super().__init__(config)
        self.streaming = False
        self.max_window_length = max_window_length  # crnn output size
        self.T = (max_window_length - 1) * config.stride[1] + config.kernel_size[1]  # melspec size
        self.slide = config.stride[1]
        self.spec_buffer = None
        self.crnn_buffer = None

    def stream_on(self):
        self.spec_buffer = torch.Tensor([]).to(self.conv[0].weight.device)
        self.crnn_buffer = torch.Tensor([]).to(self.conv[0].weight.device)
        self.streaming = True

    def stream_off(self):
        self.spec_buffer = None
        self.crnn_buffer = None
        self.streaming = False

    def forward(self, batch, hidden=None):
        batch = batch.unsqueeze(dim=1)
        if self.streaming:
            batch = torch.cat((self.spec_buffer, batch), 3)

        conv_output = self.conv(batch).transpose(-1, -2)
        gru_output, hidden = self.gru(conv_output, hidden)
        if self.streaming:
            self.spec_buffer = batch[:, :, :, self.slide * conv_output.size(1):]
            gru_output = torch.cat((self.crnn_buffer, gru_output), 1)
            gru_output = gru_output[:, max(gru_output.size(1) - self.max_window_length, 0):]
            self.crnn_buffer = gru_output

        contex_vector = self.attention(gru_output)
        logits = self.classifier(contex_vector)
        return (logits, hidden) if self.streaming else logits
