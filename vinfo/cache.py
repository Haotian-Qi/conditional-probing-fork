import os
import re
import time
from pathlib import Path

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
        if self.available:
            print(f"Cache lock file at {self.path} was already released")
            self.acquired = False
            return
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
        self.reader = None
        self.writer = None
        self.dataset_path = dataset_path
        self.path = self.get_cache_path(dataset_path, task_name)
        self.lock = _CacheLock(self.path)
        self.force_read_cache = force_read_cache

    @staticmethod
    def _sanitize_task_name(task_name):
        return re.sub("[^A-Za-z0-9_-]", "_", task_name)

    @classmethod
    def get_cache_path(cls, dataset_path, task_name):
        task_name = cls._sanitize_task_name(task_name)
        return f"{dataset_path}.cache.{task_name}.hdf5"

    def status(self):
        if self.force_read_cache:
            # Force trying to read from the cache
            print("Forcing trying to read cache, even if not there")
            return True, False

        if not os.path.exists(self.path):
            # No cache exists, so cache is writable and can be read from immediately
            print("Cache does not exist, so is readable/writable")
            return True, True

        dataset_time = os.path.getmtime(self.dataset_path)
        cache_time = os.path.getmtime(self.path)
        if cache_time < dataset_time:
            # Cache is older than data, so erase cache and write from scratch
            os.remove(self.path)
            print(f"Deleting old cache at {self.path}")

            # Remove lock path
            self.lock.remove()
            return True, True

        # Cache is locked by another process writing to it but can be read from.
        if not self.lock.available:
            print("Cache lock is unavailable, so cannot be used for writing")
            return True, False

        # Cache is valid, can be used for reading and writing
        return True, True

    def open(self, mode="r"):
        """
        Opens the cache file with the specified mode.

        This operation is idempotent; if the file has already been opened,
        it will do nothing.

        The cache uses a single-writer, multiple-reader model,
        so opening the file in a reading and writing mode opens two file handles,
        one each for reading and writing.

        The valid values of `mode` are `'rw'` and `'r'`
        """
        if mode not in ("rw", "r"):
            raise ValueError(f"Invalid mode {mode!r}. Mode must be one of: 'rw', 'r'")
        if mode == "rw" and self.writer is None:
            self.lock.acquire()
            self.writer = h5py.File(self.path, "a", libver="latest")
            self.writer.swmr_mode = True
        if self.reader is None:
            self.reader = h5py.File(self.path, "r", libver="latest", swmr=True)

    def close(self):
        """
        Closes the cache file.

        This operation is idempotent; if the file has already been closed,
        it will do nothing.
        """
        if self.reader is not None:
            self.reader.close()
            self.reader = None
            self.lock.release()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def write(self, path, data):
        """
        Writes to cache if it is open.
        """
        if self.writer is None:
            raise RuntimeError("Cache file is not open for writing.")
        if not self.lock.acquired:
            raise RuntimeError("Cache lock file has not been acquired")
        dataset = self.writer.create_dataset(path, data.shape)
        dataset[:] = data
        dataset.flush()

    def read(self, path):
        """
        Reads from the cache file
        """
        if self.reader is None:
            raise RuntimeError("Cache file is not opening for reading.")
        return self.reader[path]

    def __contains__(self, x):
        if self.reader is None:
            raise RuntimeError("Cache file reader is closed.")
        return x in self.reader
