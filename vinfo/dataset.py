import h5py
import Levenshtein as levenshtein
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
    get_split_dictionary,
)
from yaml import YAMLObject

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
        self.data = get_split_dictionary()

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
        for sentence in tqdm(
            self.data_loader.yield_dataset(split), desc="[loading]"
        ):
            input_tensors = []
            for dataset in self.input_datasets:
                input_tensors.append(dataset.tensor_of_sentence(sentence, split))
            output_tensor = self.output_dataset.tensor_of_sentence(
                sentence, split
            )
            yield (input_tensors, output_tensor, sentence)

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


class ELMoData(InitYAMLObject):
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


class HuggingfaceData(InitYAMLObject):
    """
    Loading and serving minibatches of tokens to input
    to a Huggingface-loaded model.
    """

    yaml_tag = "!HuggingfaceData"

    def __init__(self, args, model_string, cache=None, wait_for_cache=False):
        print("Constructing HuggingfaceData of {}".format(model_string))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_string
        )  # , add_prefix_space=True)
        self.args = args
        self.cache = cache
        self.cache_writers = get_split_dictionary()
        self.cache_tokens = get_split_dictionary()
        self.cache_alignments = get_split_dictionary()
        self.cache_is_setup = False

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

    def _setup_cache(self):
        """
        Constructs readers for caches that exist
        and writers for caches that do not.
        """
        if self.cache is None or self.cache_is_setup:
            return

        # Check cache readable/writeable
        for split in SPLITS:
            path, readable, writable = self.cache.get_cache_path_and_check(
                split, "hfacetokens"
            )

            if not readable and not writable:
                self.cache = None
                print(
                    f"Not using the cache, as {split} cache is neither readable nor writable."
                )
                return

            if readable:
                # Load from cache
                f = h5py.File(path, "r")
                self.cache_tokens[split] = (
                    torch.tensor(f[str(i) + "tok"][()]) for i in range(len(f.keys()))
                )
                self.cache_alignments[split] = (
                    torch.tensor(f[str(i) + "aln"][()]) for i in range(len(f.keys()))
                )
            elif writable:
                # Setup writer
                self.cache_writers[split] = self.cache.get_hdf5_cache_writer(path)
                self.cache_tokens[split] = None
                self.cache_alignments[split] = None
            else:
                # Should not reach here
                raise ValueError(
                    f"{split.title()} cache neither readable nor writeable"
                )

        self.cache_is_setup = True

    def tensor_of_sentence(self, sentence, split):
        self._setup_cache()
        if split not in SPLITS:
            raise ValueError("Unknown split: {}".format(split))

        # Cache is not being used
        if self.cache is None:
            return self._tensor_of_sentence(sentence, split)

        # Cache is being read from
        if self.cache_tokens[split] is not None:
            return next(self.cache_tokens[split]), next(self.cache_alignments[split])

        # Get tensor of sentence, and write to cache
        cache_writer = self.cache_writers[split]
        wordpiece_indices, alignments = self._tensor_of_sentence(sentence, split)

        tok_string_key = (
            str(len(list(filter(lambda x: "tok" in x, cache_writer.keys())))) + "tok"
        )
        tok_dset = cache_writer.create_dataset(tok_string_key, wordpiece_indices.shape)
        tok_dset[:] = wordpiece_indices

        aln_string_key = (
            str(len(list(filter(lambda x: "aln" in x, cache_writer.keys())))) + "aln"
        )
        aln_dset = cache_writer.create_dataset(aln_string_key, alignments.shape)
        aln_dset[:] = alignments

        return wordpiece_indices, alignments

    def _tensor_of_sentence(self, sentence, split):
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


class AnnotationData(InitYAMLObject):
    """
    Loading and serving minibatches of data from annotations
    """

    yaml_tag = "!AnnotationDataset"

    def __init__(self, args, task):
        self.args = args
        self.task = task
        # self.task.setup_cache()

    def tensor_of_sentence(self, sentence, split_string):
        """
        Converts from a tuple-formatted sentence (e.g, from conll-formatted data)
        to a Torch tensor of integers representing the annotation
        """
        alignment = torch.eye(len(sentence))
        return self.task.labels_of_sentence(sentence, split_string), alignment


class Loader(InitYAMLObject):
    """
    Base class for objects that read datasets from disk
    and yield sentence buffers for tokenization and labeling
    Strictly for description
    """

    yaml_tag = "!Loader"


class OntonotesReader(Loader):
    """
    Minutae for reading the Ontonotes dataset,
    as formatted as described in the readme
    """

    yaml_tag = "!OntonotesReader"

    def __init__(self, args, train_path, dev_path, test_path, cache):
        print("Constructing OntoNotesReader")
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.cache = cache

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

    def yield_dataset(self, split_string):
        """
        Yield a list of attribute lines, given by ontonotes_fields,
        for each sentence in the training set of ontonotes
        """
        path = (
            self.train_path
            if split_string == TRAIN_STR
            else (
                self.dev_path
                if split_string == DEV_STR
                else (self.test_path if split_string == TEST_STR else None)
            )
        )
        if path is None:
            raise ValueError("Unknown split string: {}".format(split_string))

        with open(path) as fin:
            for sentence in OntonotesReader.sentence_lists_of_stream(fin):
                yield sentence


class SST2Reader(Loader):
    """
    Minutae for reading the Stanford Sentiment (SST-2)
    dataset, as downloaded from the GLUE website.
    """

    yaml_tag = "!SST2Reader"

    def __init__(self, args, train_path, dev_path, test_path, cache):
        print("Constructing SST2Reader")
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.cache = cache

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

    def yield_dataset(self, split_string):
        """
        Yield a list of attribute lines, given by ontonotes_fields,
        for each sentence in the training set of ontonotes
        """
        path = (
            self.train_path
            if split_string == TRAIN_STR
            else (
                self.dev_path
                if split_string == DEV_STR
                else (self.test_path if split_string == TEST_STR else None)
            )
        )
        if path is None:
            raise ValueError("Unknown split string: {}".format(split_string))

        with open(path) as fin:
            for sentence in SST2Reader.sentence_lists_of_stream(fin):
                yield sentence
