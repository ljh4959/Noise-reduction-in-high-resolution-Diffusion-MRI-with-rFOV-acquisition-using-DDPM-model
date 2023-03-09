from dataclasses import dataclass


@dataclass
class TrainingConfig:
    device = "cuda"


training_config = TrainingConfig()