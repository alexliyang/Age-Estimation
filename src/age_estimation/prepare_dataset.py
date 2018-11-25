from params import get_args
from data_utils.preparators import WikiDatasetPreparator


if __name__ == '__main__':
    args = get_args('prepare_dataset')

    try:
        print('\n[INFO] Downloading and unzipping {0}...'.format(args.dataset_name))
        wiki_dataset_preparator = WikiDatasetPreparator(
            fname=args.fname,
            origin=args.origin,
            file_hash=args.file_hash,
            dataset_name = args.dataset_name
        )
        # wiki_dataset_preparator.download()

        print('\n[INFO] Dataset preparation ...'.format(args.dataset_name))
        wiki_dataset_preparator.prepare()

        print('[INFO] Success')

    except Exception as e:
        print('An exception occurred: {0}'.format(e))