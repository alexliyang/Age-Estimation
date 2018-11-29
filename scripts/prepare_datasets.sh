#!/usr/bin/env bash

cd ../src/age_estimation/
python prepare_dataset.py \
--dataset_name 'WIKI_dataset' \
--fname 'wiki.tar.gz' \
--origin 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz' \
--file_hash '6d4b8474b832f13f7f07cfbd3ed79522'

python prepare_dataset.py \
--dataset_name 'SoF_dataset' \
--fname 'original images.rar' \
--origin 'https://drive.google.com/uc?authuser=0&id=0BwO0RMrZJCioR2FNQ3k1Z3FtODg&export=download' \
--file_hash ''

cd ../../data/
rm 'wiki.tar.gz'
rm -r 'wiki'

rm 'original images.rar'
rm -r 'original images'