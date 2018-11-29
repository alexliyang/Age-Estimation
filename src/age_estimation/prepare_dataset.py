from utils.params import get_args
from data.preparators import WikiDatasetPreparator
from data.preparators import SoFDatasetPreparator


if __name__ == '__main__':
    args = get_args('prepare_dataset')

    dataset_preparator = None

    if args.dataset_name == 'WIKI_dataset':
        dataset_preparator = WikiDatasetPreparator(
            fname=args.fname,
            origin=args.origin,
            file_hash=args.file_hash,
            dataset_name = args.dataset_name
        )

    elif args.dataset_name == 'SoF_dataset':
        dataset_preparator = SoFDatasetPreparator(
            fname=args.fname,
            origin=args.origin,
            file_hash=args.file_hash,
            dataset_name = args.dataset_name
        )
    #
    print('\n[INFO] Downloading and unzipping {0}...'.format(args.dataset_name))
    dataset_preparator.download()

    print('\n[INFO] Dataset preparation ...'.format(args.dataset_name))
    dataset_preparator.prepare()