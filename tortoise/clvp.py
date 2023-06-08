from dataclasses import dataclass

import torch.nn as nn

@dataclass
class CLVPConfig:
    ...

class CLVP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()
