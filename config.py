"""
This script defines different classes with the fields that customize a NeRF architecture or a training phase
"""

import dataclasses
from omegaconf import OmegaConf


@dataclasses.dataclass
class TrainingConfig:
    """Sub-configuration for the training procedure."""

    lr: float = 5e-4            # learning rate
    bs: int = 1024              # batch size
    workers: int = 4            # number of workers
    weight_decay: float = 0     # weight decay
    chunk: int = 32*1024        # maximum number of rays to process simultaneously, to regulate memory usage
    perturb: float = 1.0        # factor to perturb depth sampling points
    noise_std: float = 0.0      # std dev of noise added to regularize sigma
    #use_disp: bool = True       # use disparity depth sampling
    use_disp: bool = False       # True is buggy with satellite FIXME

    lr_scheduler: str = "step"


@dataclasses.dataclass
class DefaultConfig:
    """Default NeRF configuration."""

    name: str = "nerf"
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    layers: int = 8           # number of fully connected layers in the shared structure
    feat: int = 256           # number of hidden units in the fully connected layers
    mapping: bool = True      # use positional encoding
    siren: bool = False       # use Siren activation if True, otherwise use ReLU
    n_samples: int = 64       # number of coarse samples
    #n_importance: int = 64    # number of additional fine samples for the fine model
    n_importance: int = 0    # number of additional fine samples for the fine model
    variant: str = "classic"

    # skip connections
    skips: list = dataclasses.field(default_factory=lambda: [4])
    # input sizes of the spatial (xyz) and viewing direction (dir) vectors
    input_sizes: list = dataclasses.field(default_factory=lambda: [3, 0])
    # number of frequencies to use in positional encoding for xyz and dir
    mapping_sizes: list = dataclasses.field(default_factory=lambda: [10, 4])


@dataclasses.dataclass
class SNerfBasicConfig:
    """S-NeRF configuration."""

    name: str = "s-nerf"
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    layers: int = 8
    feat: int = 256 #100
    mapping: bool = True
    siren: bool = False #True
    n_samples: int = 64
    n_importance: int = 0
    variant: str = "s-nerf"
    skips: list = dataclasses.field(default_factory=lambda: [4])
    input_sizes: list = dataclasses.field(default_factory=lambda: [3, 0])
    mapping_sizes: list = dataclasses.field(default_factory=lambda: [10, 4])


def load_config(args):

    config_dict = {"nerf": DefaultConfig, "s-nerf": SNerfBasicConfig}

    conf = OmegaConf.structured(config_dict[args.config_name])

    #if "s-nerf" in args.config_name:
        #conf.training.lr = float(1e-4)
        #conf.training.bs = int(256)

    if args.dataset_name == "blender":
        conf.input_sizes[1] = 3

    return conf
