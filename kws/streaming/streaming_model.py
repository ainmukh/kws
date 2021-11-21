from kws.base import CRNN
import torch


class CRNN_streaming(CRNN):
    def __init__(self, config: TaskConfig, max_window_length: int = 7):
        super().__init__(config)
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


