import torch
import torch.nn as nn
import torch.nn.functional as F


# Pay attention to _groups_ param
def SepConv(in_size, out_size, kernel_size, stride, padding=0):
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size[1],
                        stride=stride[1], groups=in_size,
                        padding=padding),

        torch.nn.Conv1d(in_size, out_size, kernel_size=1,
                        stride=stride[0], groups=int(in_size / kernel_size[0]))
    )


class CRNN(nn.Module):

    def __init__(self, config):
        super(CRNN, self).__init__()

        self.sepconv = SepConv(in_size=config.n_mels, out_size=config.hidden_size,
                               kernel_size=config.kernel_size, stride=config.stride)

        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size,
                          num_layers=config.gru_num_layers,
                          dropout=0.1,
                          bidirectional=config.bidirectional)

    def forward(self, x, hidden):
        x = self.sepconv(x)

        # (BS, hidden, seq_len) ->(seq_len, BS, hidden)
        x = x.permute(2, 0, 1)
        x, hidden = self.gru(x, hidden)
        # x : (seq_len, BS, hidden * num_dirs)
        # hidden : (num_layers * num_dirs, BS, hidden)

        return x, hidden


class AttnMech(nn.Module):

    def __init__(self, config):
        super(AttnMech, self).__init__()

        ratio = 2 if config.bidirectional else 1
        lin_size = config.hidden_size * ratio

        self.Wx_b = nn.Linear(lin_size, lin_size)
        self.Vt = nn.Linear(lin_size, 1, bias=False)

    def forward(self, inputs, data=None):

        # count only 1 e_t
        if data is None:
            x = inputs
            x = torch.tanh(self.Wx_b(x))
            e = self.Vt(x)
            return e

        # recount attention for full vector e
        e = inputs
        # (BS, seq_len, hid_size*num_dirs)
        data = data.transpose(0, 1)
        alphas = F.softmax(e, dim=-1).unsqueeze(1)
        c = torch.matmul(alphas, data).squeeze()   # attetntion_vector
        return c


class FullModel(nn.Module):

    def __init__(self, config, CRNN_model, attn_layer, max_window_length: int = None, t: int = None):
        super(FullModel, self).__init__()

        self.CRNN_model = CRNN_model
        self.attn_layer = attn_layer

        # ll_in_size, ll_out_size = HIDDEN_SIZE * GRU_NUM_DIRS, NUM_CLASSES
        # last layer
        ratio = 2 if config.bidirectional else 1
        self.U = nn.Linear(config.hidden_size * ratio,
                           config.num_classes, bias=False)

        # streaming
        self.streaming = False
        self.max_window_length = max_window_length
        self.slide = config.stride + t
        self.buffer = None

    def stream_on(self):
        self.streaming = True
        self.buffer = torch.Tensor([])

    def stream_off(self):
        self.streaming = False

    def forward(self, batch, hidden=None):
        if self.streaming:
            self.buffer = torch.cat((self.buffer, batch))[:, :, self.slide:]

        output, hidden = self.CRNN_model(batch, hidden)
        # output : (seq_len, BS, hidden * num_dirs)
        # hidden : (num_layers * num_dirs, BS, hidden)

        e = []
        for seq_el in output:
            e_t = self.attn_layer(seq_el)  # (BS, 1)
            e.append(e_t)
        e = torch.cat(e, dim=1)           # (BS, seq_len)

        c = self.attn_layer(e, output)    # attention_vector
        Uc = self.U(c)
        return Uc, hidden               # we will need to get probs, so we use return logits
