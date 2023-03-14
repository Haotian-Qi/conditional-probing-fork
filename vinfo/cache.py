import glob
import itertools
import logging
import os
from pathlib import Path
import time

import h5py
from utils import DEV_STR, SPLITS, TEST_STR, TRAIN_STR, InitYAMLObject
from yaml import YAMLObject


class WholeDatasetCache(InitYAMLObject):
    """
    Class for managing the storage and recall of
    precomputed featurized versions of datasets and annotations
    """

    cache_checked = False
    yaml_tag = "!WholeDatasetCache"

    def __init__(
        self,
        train_path,
        dev_path,
        test_path,
        force_read_cache=False,
        wait_for_cache=False,
    ):
        self.paths = {
            TRAIN_STR: train_path,
            DEV_STR: dev_path,
            TEST_STR: test_path,
        }
        self.force_read_cache = force_read_cache
        self.wait_for_cache = wait_for_cache
        self.waiting_for_cache = False

    @staticmethod
    def _sanitize_task_name(task_name):
        return "".join(
            [c for c in task_name if c.isalpha() or c.isdigit() or c == " "]
        ).rstrip()

    @staticmethod
    def _wait_for_lock(lock_path):
        """
        Waits for the file at `lock_path` to be released, i.e. deleted.
        """
        while os.path.exists(lock_path):
            time.sleep(1)

    def _get_cache_path(self, split, task_name):
        return f"{self.paths[split]}.cache.{task_name}.hdf5"

    def get_cache_path_and_check(self, split, task_name):
        """Provides the path for cache files, and cache validity

        Arguments:
          split: {TRAIN_STR, DEV_STR, TEST_STR} determining data split
          task_name: unique identifier for task/annotation type
        Returns:
          - filesystem path for the cache
          - bool True if the cache is valid to be read from
            (== exists and no lock file exists indicating that it is
             being written to. Does not solve race conditions; use
             cache with caution.)
        """
        if split not in SPLITS:
            return ValueError("Unknown split name: {}".format(split))
        task_name = self._sanitize_task_name(task_name)
        cache_path = self._get_cache_path(split, task_name)
        read_cache, write_cache = self.check_cache(split, cache_path)
        print(
            "For task {}, split {}, we are reading:{}, writing:{} the cache".format(
                task_name, split, read_cache, write_cache
            )
        )
        return cache_path, read_cache, write_cache

    def check_cache(self, split, cache_path):
        """
        Determines whether caches can be read from our written to.

        The dataset at `path` must exist, or an error is raised.

        If the caches of the dataset for the specified `task_name`
        in the form `${path}*.cache` are older than the file at `${path}`,
        then all caches are erased.

        If the cache's lock file is present, then the cache is not available
        for reading or writing. If `self.wait_for_cache` is True,
        then the method blocks until the lock file is released.

        Arguments:
          path: The full disk path to a dataset
        Outputs:
          (read_cache, write_cache):
            read_cache True if cache should be used for reading, False otherwise
            write_cache True if cache should be written to, False otherwise
        """
        dataset_path = self.paths[split]
        dataset_time = os.path.getmtime(dataset_path)
        if self.force_read_cache:
            # Force trying to read from the cache
            print("Forcing trying to read cache, even if not there")
            return True, False
        if not os.path.exists(cache_path):
            # No cache exists; write one
            return False, True

        lock_path = cache_path + ".lock"
        cache_time = os.path.getmtime(cache_path)
        if cache_time < dataset_time:
            # Cache is older than data, so erase cache
            os.remove(cache_path)
            logging.info("Cache erased at: {}".format(cache_path))
            if os.path.exists(lock_path):
                os.remove(lock_path)
            return False, True
        if os.path.exists(lock_path):
            # Cache locked, being written to by this or another process
            if self.wait_for_cache:
                print(f"Waiting for {lock_path} to be released...")
                self._wait_for_lock(lock_path)
                return self.check_cache(split, cache_path)
            return False, False
        return True, False  # Cache is valid; read from it

    def release_locks(self):
        """Removes lock files from caches"""
        lock_paths = (f"{path}.lock" for path in self.paths.values())
        for cache_lock_path in lock_paths:
            print("Removing cache lock file at {}".format(cache_lock_path))
            os.remove(cache_lock_path)

    def get_hdf5_cache_writer(self, cache_path):
        """Gets an hdf5 file writer and makes a lock file for it"""
        print("Getting cache writer for {}".format(cache_path))
        Path(cache_path + ".lock").touch()
        return h5py.File(cache_path, "a")

    # def is_valid():
    #  """
    #  Getter method for whether the data cached is valid to be used again

    #  Returns:
    #    True if Task models can reuse
    #  """
    #  if not self.cache_checked:
    #    raise Exception("Cache has not been checked but is being queried")
