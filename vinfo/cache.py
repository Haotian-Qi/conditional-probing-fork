import logging
import os
from pathlib import Path
import re
import time

import h5py
from utils import InitYAMLObject
from yaml import YAMLObject


class _CacheLock:
    def __init__(self, cache_path):
        self.path = cache_path + ".lock"
        self.acquired = False

    @property
    def available(self):
        return not os.path.exists(self.path)

    def acquire(self):
        if self.acquired:
            return
        while not self.available:
            pass
        print(f"Acquiring cache lock file at {self.path}")
        try:
            Path(self.path).touch()
        except OSError:
            pass
        else:
            self.acquired = True

    def release(self):
        if not self.acquired:
            return
        if not self.available:
            print(f"Cache lock file at {self.path} was already released")
        print(f"Releasing cache lock file at {self.path}")
        try:
            os.remove(self.path)
        except OSError:
            pass
        else:
            self.acquired = False

    def remove(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def wait(self):
        """
        Waits for the file at cache lock path to be released, i.e. deleted.
        """
        print(f"Waiting for cache lock file at {self.path}")
        while os.path.exists(self.path):
            time.sleep(1)


class WholeDatasetCache(InitYAMLObject):
    """
    Class for managing the storage and recall of
    precomputed featurized versions of datasets and annotations
    """

    yaml_tag = "!WholeDatasetCache"

    def __init__(
        self,
        dataset_path,
        task_name,
        force_read_cache=False,
    ):
        self.file = None
        self.dataset_path = dataset_path
        self.cache_path = self.get_cache_path(dataset_path, task_name)
        self.lock = _CacheLock(self.cache_path)
        self.force_read_cache = force_read_cache

    @staticmethod
    def _sanitize_task_name(task_name):
        return re.sub("[^A-Za-z0-9_-]", "_", task_name)

    @classmethod
    def get_cache_path(cls, dataset_path, task_name):
        task_name = cls._sanitize_task_name(task_name)
        return f"{dataset_path}.cache.{task_name}.hdf5"

    def status(self):
        # Cache is locked by another process writing to it
        if not self.lock.available and not self.lock.acquired:
            return False, False

        dataset_time = os.path.getmtime(self.dataset_path)
        if self.force_read_cache:
            # Force trying to read from the cache
            print("Forcing trying to read cache, even if not there")
            return True, False
        if not os.path.exists(self.cache_path):
            # No cache exists, so cache is writable
            return False, True

        cache_time = os.path.getmtime(self.cache_path)
        if cache_time < dataset_time:
            # Cache is older than data, so erase cache and write from scratch
            os.remove(self.cache_path)
            logging.info("Cache erased at: {}".format(self.cache_path))

            # Remove lock path
            self.lock.remove()
            return False, True

        # Cache is valid, can be used for reading and writing
        return True, True

    def open(self):
        """
        Opens the cache file for appending.

        This operation is idempotent; if the file has already been opened,
        it will do nothing.
        """
        if self.file is not None:
            return
        self.file = h5py.File(self.cache_path, "a")

    def close(self):
        """
        Closes the cache file.

        This operation is idempotent; if the file has already been closed,
        it will do nothing.
        """
        if self.file is None:
            return
        self.file.close()
        self.file = None

    def write(self, path, data):
        """
        Writes to cache if it is open.
        """
        if self.file is None:
            raise RuntimeError("Cache file is closed.")
        if not self.lock.acquired:
            raise RuntimeError("Cache lock file has not been acquired")
        dataset = self.file.create_dataset(path, data.shape)
        dataset[:] = data

    def read(self, path):
        """
        Reads from the cache file
        """
        if self.file is None:
            raise RuntimeError("Cache file is closed.")
        return self.file[path]

    def __contains__(self, x):
        if self.file is None:
            raise RuntimeError("Cache file is closed.")
        return x in self.file
