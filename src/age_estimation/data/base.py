import os
import six
import shutil
import tarfile
import pyunpack
import zipfile
from keras.utils import get_file


class BaseDatasetPreparator(object):

    def __init__(self, archive_name, url, hash, dataset_name, archive_format='auto', data_dir=os.path.abspath('../../data')):
        self.archive_name = archive_name
        self.url = url
        self.hash = hash
        self.dataset_name = dataset_name

        self.archive_format = archive_format
        self.data_dir = data_dir
        self.unzip_dir = None

    def download(self):
        archive_path = get_file(
            fname=self.archive_name,
            origin=self.url,
            cache_subdir=self.data_dir,
            file_hash=self.hash,
            extract=False
        )
        self.unzip_dir = os.path.join(archive_path[:-len(self.archive_name)], self.archive_name.split('.')[0])
        self._extract_archive(archive_path=archive_path, unzip_dir=self.unzip_dir, archive_format=self.archive_format)

        return self.unzip_dir

    def prepare(self):
        raise NotImplementedError

    def _create_own_dir(self):
        prepared_dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(prepared_dataset_dir):
            os.makedirs(prepared_dataset_dir)
            os.makedirs(os.path.join(prepared_dataset_dir, 'images'))
            return True

        return False

    def _extract_archive(self, archive_path, unzip_dir='.', archive_format='auto'):
        if archive_format is None:
            return False
        if archive_format == 'auto':
            archive_format = ['tar', 'zip', 'rar']
        if isinstance(archive_format, six.string_types):
            archive_format = [archive_format]

        is_match_fn = None
        open_fn = None

        for archive_type in archive_format:
            if archive_type == 'tar':
                open_fn = tarfile.open
                is_match_fn = tarfile.is_tarfile

            if archive_type == 'zip':
                open_fn = zipfile.ZipFile
                is_match_fn = zipfile.is_zipfile

            if archive_type == 'rar':
                archive = pyunpack.Archive(archive_path)
                archive.extractall('\\'.join(unzip_dir.split('\\')[:-1]))
                return True

            if is_match_fn(archive_path):
                with open_fn(archive_path) as archive:
                    try:
                        archive.extractall(unzip_dir)
                    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                        if os.path.exists(unzip_dir):
                            if os.path.isfile(unzip_dir):
                                os.remove(unzip_dir)
                            else:
                                shutil.rmtree(unzip_dir)
                        raise
                return True

        return False