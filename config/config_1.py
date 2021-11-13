import torch
import dataclasses


@dataclasses.dataclass
class TaskConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 30
    n_mels: int = 40
    kernel_size: int = 16
    stride: int = 10
    hidden_size: int = 64
    gru_num_layers: int = 1
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    save_to: str = 'saved/model1.pth',
    temperature: int = 2
