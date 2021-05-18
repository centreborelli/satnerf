from .satellite import SatelliteDataset

def load_dataset(args, split):

    dataset_dict = {'satellite': SatelliteDataset}

    dataset = dataset_dict[args.dataset_name]
    return dataset(root_dir=args.root_dir,
                   img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                   split=split,
                   cache_dir=args.cache_dir)
