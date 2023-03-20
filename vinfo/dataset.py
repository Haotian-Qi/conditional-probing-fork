from abc import ABCMeta, abstractmethod
import hashlib

import Levenshtein as levenshtein
import numpy as np
import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import (
    DEV_STR,
    SPLITS,
    TEST_STR,
    TRAIN_STR,
    InitYAMLObject,
    check_split,
    new_split_dictionary,
)
from yaml import YAMLObject, YAMLObjectMetaclass

BATCH_SIZE = 50
"""
Classes for loading, caching, and yielding text datasets
"""


# class Dataset(Dataset, InitYAMLObject):
#  """
#  Base class for objects that serve batches of
#  tensors. For decoration/explanation only
#  """
#  yaml_tag = '!Dataset'


class IterableDatasetWrapper(Dataset):  # (IterableDataset):
    """
    Wrapper class to pass to a DataLoader so it doesn't
    think the underlying generator should have a len() fn.

    But I gave up on this for various reasons so it's just
    a normal dataset, here in case I try again.
    """

    def __init__(self, generator):
        self.generator = generator  # [x for x in generator]

    def __iter__(self):
        return iter(self.generator)

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        return self.generator[idx]


class ListDataset(Dataset, InitYAMLObject):
    """
    Container class for collecting multiple annotation or
    representation datasets and a single target task dataset
    , and serving all of them
    """

    yaml_tag = "!ListDataset"

    def __init__(self, args, data_loader, output_dataset, input_datasets):
        """
        Arguments:
          output_datset:
        """
        self.args = args
        self.input_datasets = input_datasets
        self.output_dataset = output_dataset
        self.data_loader = data_loader
        self.data = new_split_dictionary()

    def get_train_dataloader(self, shuffle=True):
        """Returns a PyTorch DataLoader object with the training data"""
        return self._get_dataloader(TRAIN_STR, shuffle)

    def get_dev_dataloader(self, shuffle=False):
        """Returns a PyTorch DataLoader object with the dev data"""
        return self._get_dataloader(DEV_STR, shuffle)

    def get_test_dataloader(self, shuffle=False):
        return self._get_dataloader(TEST_STR, shuffle)

    def _get_dataloader(self, split, shuffle):
        if self.data[split] is None:
            self.data[split] = list(self.load_data(split))
        generator = IterableDatasetWrapper(self.data[split])
        return DataLoader(
            generator,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def load_data(self, split):
        """Loads data from disk into RAM tensors for passing to a network on GPU

        Iterates through the training set once, passing each sentence to each
        input Dataset and the output Dataset
        """
        for dataset in self.input_datasets:
            dataset.before_load()
        for sentence in tqdm(self.data_loader.yield_dataset(split), desc="[loading]"):
            input_tensors = []
            for dataset in self.input_datasets:
                input_tensors.append(dataset.tensor_of_sentence(sentence, split))
            output_tensor = self.output_dataset.tensor_of_sentence(sentence, split)
            yield (input_tensors, output_tensor, sentence)
        for dataset in self.input_datasets:
            dataset.after_load()

    def collate_fn(self, observation_list):
        """
        Combines observations (input_tensors, output_tensor, sentence) tuples
        input_tensors is of the form ((annotation, alignment), ..., (annotation, alignment))
        output_tensor is of the form (annotation, alignment),

        to batches of observations ((batches_input_1, batches_input_2), batches_output, sentences)
        """
        sentences = (x[2] for x in observation_list)
        max_corpus_token_len = max((len(x) for x in sentences))
        input_annotation_tensors = []
        input_alignment_tensors = []
        input_tensor_count = len(observation_list[0][0])
        for input_tensor_index in range(input_tensor_count):
            max_annotation_token_len = max(
                [x[0][input_tensor_index][0].shape[0] for x in observation_list]
            )
            intermediate_annotation_list = []
            intermediate_alignment_list = []
            for input_annotation, input_alignment in (
                (x[0][input_tensor_index][0], x[0][input_tensor_index][1])
                for x in observation_list
            ):
                if len(input_annotation.shape) == 1:  # word-level ids
                    new_annotation_tensor = torch.zeros(
                        max_annotation_token_len, dtype=torch.long
                    )
                    new_annotation_tensor[: len(input_annotation)] = input_annotation
                elif len(input_annotation.shape) == 2:  # characeter-level ids
                    new_annotation_tensor = torch.zeros(
                        max_annotation_token_len, input_annotation.shape[1]
                    ).long()
                    new_annotation_tensor[: len(input_annotation), :] = input_annotation
                intermediate_annotation_list.append(new_annotation_tensor)
                new_alignment_tensor = torch.zeros(
                    max_annotation_token_len, max_corpus_token_len
                )
                new_alignment_tensor[
                    : input_alignment.shape[0], : input_alignment.shape[1]
                ] = input_alignment
                intermediate_alignment_list.append(new_alignment_tensor)
            input_annotation_tensors.append(
                torch.stack(intermediate_annotation_list).to(self.args["device"])
            )
            input_alignment_tensors.append(
                torch.stack(intermediate_alignment_list).to(self.args["device"])
            )

        intermediate_annotation_list = []
        intermediate_alignment_list = []
        max_output_annotation_len = max([x[1][0].shape[0] for x in observation_list])
        for output_annotation, output_alignment in (x[1] for x in observation_list):
            new_annotation_tensor = torch.zeros(
                max_output_annotation_len, dtype=torch.long
            )
            new_annotation_tensor[: len(output_annotation)] = output_annotation
            intermediate_annotation_list.append(new_annotation_tensor)
        output_annotation_tensor = torch.stack(intermediate_annotation_list).to(
            self.args["device"]
        )
        sentences = [x[2] for x in observation_list]
        return (
            (input_annotation_tensors, input_alignment_tensors),
            output_annotation_tensor,
            sentences,
        )


class BaseDataMeta(ABCMeta, YAMLObjectMetaclass):
    pass


class BaseData(InitYAMLObject, metaclass=BaseDataMeta):
    def before_load(self):
        pass

    @abstractmethod
    def tensor_of_sentence(self, sentence, split):
        pass

    def after_load(self):
        pass


class ELMoData(BaseData):
    """
    Loading and serving minibatches of tokens to input to
    ELMo, as mediated by allennlp.
    """

    yaml_tag = "!ELMoData"

    def __init__(self, args):
        self.args = args

    def tensor_of_sentence(self, sentence, split_string):
        """
        Provides character indices for a single sentence.
        """
        words = [x[1] for x in sentence]
        alignment = torch.eye(len(words))
        return batch_to_ids([words])[0, :, :], alignment
        # for index, token in enumerate([x[1] for x in sentence]):


class HuggingfaceData(BaseData):
    """
    Loading and serving minibatches of tokens to input
    to a Huggingface-loaded model.
    """

    yaml_tag = "!HuggingfaceData"

    def __init__(self, args, tokenizer_model_string, caches, wait_for_cache=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_string)
        self.args = args
        self.caches = caches
        self.cache_statuses = new_split_dictionary()
        self.wait_for_cache = wait_for_cache

    def levenshtein_matrix(self, string1, string2):
        opcodes = levenshtein.opcodes(string1, string2)
        mtx = torch.zeros(len(string1), len(string2))
        # cumulative = 0
        for opcode in opcodes:
            opcode_type, str1b, str1e, str2b, str2e = opcode
            str1b -= 1
            str1e -= 1
            str2b -= 1
            str2e -= 1
            if opcode_type in {"equal", "replace"}:
                diff = str1e - str1b
                for i in range(diff):
                    mtx[str1b + i, str2b + i] = 1
            if opcode_type == "delete":
                diff = str1e - str1b
                for i in range(diff):
                    mtx[str1b + i, str2b] = 1
            if opcode_type == "insert":
                diff = str2e - str2b
                for i in range(diff):
                    mtx[str1b, str2b + i] = 1
        return mtx

    def token_to_character_alignment(self, tokens):
        ptb_sentence_length = sum((len(tok) for tok in tokens))
        ptb_string_token_alignment = []
        cumulative = 0
        for token in tokens:
            new_alignment = torch.zeros(ptb_sentence_length)
            for i, char in enumerate(token):
                if char == " ":
                    continue
                new_alignment[i + cumulative] = 1
            new_alignment = new_alignment / sum(new_alignment)
            cumulative += len(token)
            ptb_string_token_alignment.append(new_alignment)
        return torch.stack(ptb_string_token_alignment)

    def de_ptb_tokenize(self, tokens):
        tokens_with_spaces = []
        new_tokens_with_spaces = []
        ptb_sentence_length = sum((len(tok) for tok in tokens))
        token_alignments = []

        cumulative = 0
        for i, _ in enumerate(tokens):
            token = tokens[i]
            next_token = tokens[i + 1] if i < len(tokens) - 1 else "<EOS>"
            # Handle LaTeX-style quotes
            if token.strip() in {"``", "''"}:
                new_token = '"'
            elif token.strip() == "-LRB-":
                new_token = "("
            elif token.strip() == "-RRB-":
                new_token = ")"
            elif token.strip() == "-LSB-":
                new_token = "["
            elif token.strip() == "-RSB-":
                new_token = "]"
            elif token.strip() == "-LCB-":
                new_token = "{"
            elif token.strip() == "-RCB-":
                new_token = "}"
            else:
                new_token = token
            use_space = (
                token.strip() not in {"(", "[", "{", '"', "'", "``", "''"}
                and next_token.strip()
                not in {
                    "'ll",
                    "'re",
                    "'ve",
                    "n't",
                    "'s",
                    "'LL",
                    "'RE",
                    "'VE",
                    "N'T",
                    "'S",
                    '"',
                    "'",
                    "``",
                    "''",
                    ")",
                    "}",
                    "]",
                    ".",
                    ";",
                    ":",
                    "!",
                    "?",
                }
                and i != len(tokens) - 1
            )

            new_token = new_token.strip() + (" " if use_space else "")
            new_tokens_with_spaces.append(new_token)
            tokens_with_spaces.append(token)

            new_alignment = torch.zeros(ptb_sentence_length)
            for index, char in enumerate(token):
                new_alignment[index + cumulative] = 1
            # new_alignment = new_alignment / sum(new_alignment)
            for new_char in new_token:
                token_alignments.append(new_alignment)
            cumulative += len(token)
        return new_tokens_with_spaces, torch.stack(token_alignments)

    def hface_ontonotes_alignment(self, sentence):
        tokens = [x[1] for x in sentence]
        tokens = [
            x + (" " if i != len(tokens) - 1 else "") for (i, x) in enumerate(tokens)
        ]
        raw_tokens, ptb_to_deptb_alignment = self.de_ptb_tokenize(tokens)
        raw_string = "".join(raw_tokens)
        ptb_token_to_ptb_string_alignment = self.token_to_character_alignment(tokens)
        # tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
        hface_tokens = self.tokenizer.tokenize(raw_string)
        hface_tokens_with_spaces = [
            x + (" " if i != len(hface_tokens) - 1 else "")
            for (i, x) in enumerate(hface_tokens)
        ]
        hface_token_to_hface_string_alignment = self.token_to_character_alignment(
            hface_tokens_with_spaces
        )
        hface_string = " ".join(hface_tokens)
        hface_character_to_deptb_character_alignment = self.levenshtein_matrix(
            hface_string, raw_string
        )
        unnormalized_alignment = torch.matmul(
            torch.matmul(
                hface_token_to_hface_string_alignment.to(self.args["device"]),
                hface_character_to_deptb_character_alignment.to(self.args["device"]),
            ),
            torch.matmul(
                ptb_token_to_ptb_string_alignment.to(self.args["device"]),
                ptb_to_deptb_alignment.to(self.args["device"]).t(),
            ).t(),
        )
        return (
            (unnormalized_alignment / torch.sum(unnormalized_alignment, dim=0)).cpu(),
            hface_tokens,
            raw_string,
        )

    @staticmethod
    def _hash_sentence(sentence):
        hasher = hashlib.md5()
        for token in sentence:
            for value in token:
                hasher.update(str(value).encode("utf-8"))
        return hasher.hexdigest()

    def _read_from_cache(self, split, sentence):
        check_split(split)

        sentence_hash = self._hash_sentence(sentence)
        if sentence_hash not in self.caches[split]:
            return None

        dataset = self.caches[split].read(sentence_hash)
        return (
            torch.tensor(np.array(dataset["tok"])),
            torch.tensor(np.array(dataset["aln"])),
        )

    def _write_to_cache(self, split, sentence, wordpiece_indices, alignments):
        check_split(split)

        sentence_hash = self._hash_sentence(sentence)
        self.caches[split].write(f"{sentence_hash}/tok", wordpiece_indices)
        self.caches[split].write(f"{sentence_hash}/aln", alignments)

    def _calculate_tensor_of_sentence(self, sentence, split):
        alignment, wordpiece_strings, raw_string = self.hface_ontonotes_alignment(
            sentence
        )
        # add [SEP] and [CLS] empty alignments
        empty = torch.zeros(1, alignment.shape[1])
        alignment = torch.cat((empty, alignment, empty))
        # wordpiece_indices = torch.tensor(self.tokenizer(wordpiece_strings)
        wordpiece_indices = torch.tensor(
            self.tokenizer(raw_string).input_ids
        )  # , is_split_into_words=True))
        return wordpiece_indices, alignment

    def _naive_tensor_of_sentence(self, sentence, split_string):
        """
        Converts from a tuple-formatted sentence (e.g, from conll-formatted data)
        to a Torch tensor of integers representing subword piece ids for input to
        a Huggingface-formatted neural model
        """
        # CLS token given by tokenizer
        wordpiece_indices = []
        wordpiece_alignment_vecs = [torch.zeros(len(sentence))]
        # language tokens
        for index, token in enumerate([x[1] for x in sentence]):
            new_wordpieces = self.tokenizer.tokenize(token)
            wordpiece_alignment = torch.zeros(len(sentence))
            wordpiece_alignment[index] = 1
            for wordpiece in new_wordpieces:
                wordpiece_alignment_vecs.append(torch.clone(wordpiece_alignment))
            wordpiece_indices.extend(new_wordpieces)
        # SEP token given by tokenizer

        wordpiece_indices = torch.tensor(self.tokenizer.encode(wordpiece_indices))
        wordpiece_alignment_vecs.append(torch.zeros(len(sentence)))
        wordpiece_alignment_vecs = torch.stack(wordpiece_alignment_vecs)
        return wordpiece_indices, wordpiece_alignment_vecs

    def before_load(self):
        for split in SPLITS:
            cache = self.caches[split]
            readable, writable = cache.status()
            if readable:
                if writable:
                    cache.open("rw")
                else:
                    cache.open("r")
            self.cache_statuses[split] = (readable, writable)

    def tensor_of_sentence(self, sentence, split):
        check_split(split)

        cache = self.caches[split]
        readable, writable = self.cache_statuses[split]

        # Cache is being read from
        if readable:
            tensors = self._read_from_cache(split, sentence)
            if tensors is not None:
                return tensors

        # Either cache is readable but the sentence was not found,
        # or cache is not readable

        # Calculate tensors
        wordpiece_indices, alignments = self._calculate_tensor_of_sentence(
            sentence, split
        )

        if not writable and self.wait_for_cache:
            print(f"Waiting for cache write to finish at {cache.path}")
            cache.lock.wait()
            self.cache_statuses[split] = cache.status()
            return self.tensor_of_sentence(sentence, split)

        # Cache is writable, so save tensors
        if writable:
            self._write_to_cache(split, sentence, wordpiece_indices, alignments)

        return wordpiece_indices, alignments

    def after_load(self):
        for cache in self.caches.values():
            cache.lock.release()


class AnnotationData(BaseData):
    """
    Loading and serving minibatches of data from annotations
    """

    yaml_tag = "!AnnotationDataset"

    def __init__(self, args, task):
        self.args = args
        self.task = task
        # self.task.setup_cache()

    def tensor_of_sentence(self, sentence, split):
        """
        Converts from a tuple-formatted sentence (e.g, from conll-formatted data)
        to a Torch tensor of integers representing the annotation
        """
        alignment = torch.eye(len(sentence))
        return self.task.labels_of_sentence(sentence, split), alignment


class Loader(InitYAMLObject):
    """
    Base class for objects that read datasets from disk
    and yield sentence buffers for tokenization and labeling
    Strictly for description
    """


class OntonotesReader(Loader):
    """
    Minutae for reading the Ontonotes dataset,
    as formatted as described in the readme
    """

    yaml_tag = "!OntonotesReader"

    def __init__(self, train_path, dev_path, test_path):
        self.paths = {
            TRAIN_STR: train_path,
            DEV_STR: dev_path,
            TEST_STR: test_path,
        }

    @staticmethod
    def sentence_lists_of_stream(ontonotes_stream):
        """
        Yield sentences from raw ontonotes stream

        Arguments:
          ontonotes_stream: iterable of ontonotes file lines
        Yields:
          a buffer for each sentence in the stream; elements
          in the buffer are lists defined by TSV fields of the
          ontonotes stream
        """
        buf = []
        for line in ontonotes_stream:
            if line.startswith("#"):
                continue
            if not line.strip():
                yield buf
                buf = []
            else:
                buf.append([x.strip() for x in line.split("\t")])
        if buf:
            yield buf

    def yield_dataset(self, split):
        """
        Yield a list of attribute lines, given by ontonotes_fields,
        for each sentence in the training set of ontonotes
        """
        check_split(split)
        path = self.paths[split]

        with open(path) as fin:
            for sentence in OntonotesReader.sentence_lists_of_stream(fin):
                yield sentence


class SST2Reader(Loader):
    """
    Minutae for reading the Stanford Sentiment (SST-2)
    dataset, as downloaded from the GLUE website.
    """

    yaml_tag = "!SST2Reader"

    def __init__(self, train_path, dev_path, test_path):
        self.paths = {
            TRAIN_STR: train_path,
            DEV_STR: dev_path,
            TEST_STR: test_path,
        }

    @staticmethod
    def sentence_lists_of_stream(sst2_stream):
        """
        Yield sentences from raw sst2 stream

        Arguments:
          sst2_stream: iterable of sst2_stream lines
        Yields:
          a buffer for each sentence in the stream;
          elements in the buffer are lists defined by TSV
          fields of the ontonotes stream
        """
        _ = next(sst2_stream)  # Get rid of the column labels
        for line in sst2_stream:
            word_string, label_string = [x.strip() for x in line.split("\t")]
            word_tokens = word_string.split(" ")
            indices = [str(i) for i, _ in enumerate(word_tokens)]
            label_tokens = [label_string for _ in word_tokens]
            yield list(zip(indices, word_tokens, label_tokens))

    def yield_dataset(self, split):
        """
        Yield a list of attribute lines, given by ontonotes_fields,
        for each sentence in the training set of ontonotes
        """
        check_split(split)
        path = self.paths[split]

        with open(path) as fin:
            for sentence in SST2Reader.sentence_lists_of_stream(fin):
                yield sentence
