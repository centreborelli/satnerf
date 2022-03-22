from .nerf import *
from .satnerf import *
from .snerf import *


def load_model(args):
    if args.model == "nerf":
        model = NeRF(layers=args.fc_layers, feat=args.fc_units)
    elif args.model == "s-nerf":
        model = ShadowNeRF(layers=args.fc_layers, feat=args.fc_units)
    elif args.model == "sat-nerf":
        model = SatNeRF(layers=args.fc_layers, feat=args.fc_units, t_embedding_dims=args.t_embbeding_tau)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return model
