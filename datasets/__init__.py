from .satellite import SatelliteDataset
from .blender import BlenderDataset

def load_dataset(args, split):

    dataset_dict = {'satellite': SatelliteDataset,
                    'blender': BlenderDataset}

    dataset = dataset_dict[args.dataset_name]

    if args.dataset_name == 'satellite':
        return dataset(root_dir=args.root_dir,
                       img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                       split=split,
                       cache_dir=args.cache_dir,
                       img_downscale=args.img_downscale)
    else:
        return dataset(root_dir=args.root_dir, split=split)
