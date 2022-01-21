from .satellite import SatelliteDataset
from .blender import BlenderDataset

def load_dataset(args, split):

    dataset_dict = {'satellite': SatelliteDataset,
                    'blender': BlenderDataset}

    dataset = dataset_dict[args.dataset_name]

    outputs = []

    if args.dataset_name == 'satellite':
        d1 = dataset(root_dir=args.root_dir,
                     img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                     split=split,
                     cache_dir=args.cache_dir,
                     img_downscale=args.img_downscale,
                     depth=False)
        outputs.append(d1)
        if args.patches and split == 'train':
            d2 = dataset(root_dir=args.root_dir,
                         img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                         split=split,
                         cache_dir=args.cache_dir,
                         img_downscale=args.img_downscale,
                         depth=False,
                         patches=True,
                         patch_size=args.patch_size)
            outputs.append(d2)
        elif args.depth and split == 'train':
            d2 = dataset(root_dir=args.root_dir,
                         img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                         split=split,
                         cache_dir=args.cache_dir,
                         img_downscale=args.img_downscale,
                         depth=True)
            outputs.append(d2)
    else:
        outputs.append(dataset(root_dir=args.root_dir, split=split))

    return outputs
