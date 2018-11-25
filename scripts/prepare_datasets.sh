#!/usr/bin/env bash

cd ../src/age_estimation/
python prepare_dataset.py \
--dataset_name WIKI \
--fname imdb_meta.tar \
--origin https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar \
--file_hash 469433135f1e961c9f4c0304d0b5db1e