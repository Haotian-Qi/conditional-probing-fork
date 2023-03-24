import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from utils import InitYAMLObject
from yaml import YAMLObject


class _CacheLock:
    def __init__(self, cache_path):
        self.path = cache_path + ".lock"
        self.acquired = False

    @property
    def available(self):
        return not os.path.exists(self.path)

    def wait(self):
        while not self.available:
            pass

    def acquire(self, blocking=True):
        if self.acquired:
            return
        if blocking:
            while not self.available:
                pass
        else:
            raise RuntimeError("Lock file not available, and blocking turned off.")
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
        print(f"Releasing cache lock file at {self.path}")
        try:
            self.remove()
        except OSError:
            pass
        else:
            self.acquired = False

    def remove(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class HuggingfaceDataCache(InitYAMLObject):
    """
    Class for representing the cached, precomputed and featurized versions
    of datasets and annotations in HuggingFace format.
    """

    yaml_tag = "!HuggingfaceDataCache"

    def __init__(
        self,
        dataset_path,
        task_name,
        force_read_cache=False,
        wait_for_cache=False,
    ):
        self.data = defaultdict(dict)
        self.new_data_hashes = set()
        self.dataset_path = dataset_path
        self.path = self.get_cache_path(dataset_path, task_name)
        self.lock = _CacheLock(self.path)
        self.force_read_cache = force_read_cache
        self.wait_for_cache = wait_for_cache

    @staticmethod
    def _sanitize_task_name(task_name):
        return re.sub("[^A-Za-z0-9_-]", "_", task_name)

    @classmethod
    def get_cache_path(cls, dataset_path, task_name):
        task_name = cls._sanitize_task_name(task_name)
        return f"{dataset_path}.cache.{task_name}.hdf5"

    @staticmethod
    def _hash_sentence(sentence):
        hasher = hashlib.md5()
        for token in sentence:
            for value in token:
                hasher.update(f"{value}\t".encode("utf-8"))
        return hasher.hexdigest()

    def check(self):
        if not os.path.exists(self.path):
            return
        dataset_time = os.path.getmtime(self.dataset_path)
        cache_time = os.path.getmtime(self.path)
        if cache_time < dataset_time:
            # Cache is older than data, so erase cache and write from scratch
            os.remove(self.path)
            print(f"Deleting old cache at {self.path}")
            # Remove lock path
            self.lock.remove()

    def load(self):
        """
        Loads the cache from disk into memory.
        """
        if not os.path.exists(self.path):
            return
        with h5py.File(self.path, "r", libver="latest", swmr=True) as f:
            for sentence_hash in tqdm(f, desc="[loading cache]", unit="sentences"):
                group = f[sentence_hash]
                for part in ("tok", "aln"):
                    try:
                        tensor = torch.tensor(np.array(group[part]))
                    except KeyError:
                        break
                    else:
                        self.data[sentence_hash][part] = tensor

    def add(self, sentence, tok, aln):
        """
        Adds to the cache. This does not flush the new data to the cache file.
        """
        sentence_hash = self._hash_sentence(sentence)
        dct = self.data[sentence_hash]
        dct["tok"], dct["aln"] = tok, aln
        self.new_data_hashes.add(sentence_hash)

    def pop(self, sentence, default=None):
        """
        Returns the tokens and alignments as a tuple for the `sentence` in the cache,
        or `None` if it does not exist in the cache.
        """
        sentence_hash = self._hash_sentence(sentence)
        dct = self.data.pop(sentence_hash, default)
        if dct is not None:
            return dct["tok"], dct["aln"]
        return None

    def _flush(self):
        Path(self.path).touch()
        with h5py.File(self.path, "r+", libver="latest") as f:
            f.swmr_mode = True
            for sentence_hash in tqdm(
                self.new_data_hashes.copy(),
                desc="[writing cache]",
                unit="sentences",
            ):
                group = f.require_group(sentence_hash)
                for part, tensor in self.data[sentence_hash].items():
                    dataset = group.create_dataset(part, tensor.shape)
                    dataset[:] = tensor
                    dataset.flush()
                self.new_data_hashes.remove(sentence_hash)

    def flush(self):
        """
        Writes all new cache data to cache.
        """
        try:
            self.lock.acquire(blocking=self.wait_for_cache)
            self._flush()
            return True
        except RuntimeError:
            return False
        finally:
            self.lock.release()

    def close(self):
        """
        Closes the cache by clearing the data heard in the class instance.
        """
        self.flush()
        self.data.clear()
        self.new_data_hashes.clear()
