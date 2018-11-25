import argparse

def _prepare_dataset_arguments(parser):
    parser.add_argument('-d', '--dataset_name', default='WIKI_dataset')
    parser.add_argument('-f', '--fname', default='wiki.tar.gz')
    parser.add_argument('-o', '--origin', default='https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz')
    parser.add_argument('-fh', '--file_hash', default='6d4b8474b832f13f7f07cfbd3ed79522')

    return parser

def get_args(module_name):
    arguments_dict = {
        'prepare_dataset': _prepare_dataset_arguments
    }
    parser = argparse.ArgumentParser(module_name)
    parser = arguments_dict[module_name](parser)

    return parser.parse_args()