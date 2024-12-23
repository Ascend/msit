from dataclasses import dataclass
import os


@dataclass
class OmniAttentionConfig:
    pool_size: int = 100
    num_mutation: int = 10
    sink: int = 128
    recent: int = 256
    model_path: str = None
    model_name: str = None
    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("Please specify path to HF model and tokenizer.")
        if self.model_name is None:
            self.model_name = os.path.basename(os.path.abspath(self.model_path))