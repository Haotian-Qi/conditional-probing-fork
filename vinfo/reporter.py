from collections import defaultdict
import os
from yaml import YAMLObject
from utils import InitYAMLObject
from stanza.models.ner.scorer import score_by_entity

from tqdm import tqdm
#from scipy.stats import spearmanr, pearsonr
import numpy as np 
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import torch

class Reporter(InitYAMLObject):
  """Base class for reporting.

  Attributes:
    test_reporting_constraint: Any reporting method
      (identified by a string) not in this list will not
      be reported on for the test set.
  """

  def __init__(self, args, dataset):
    raise NotImplementedError("Inherit from this class and override __init__")

  def __call__(self, prediction_batches, dataloader, split_name):
    """
    Performs all reporting methods as specifed in the yaml experiment config dict.
    
    Any reporting method not in test_reporting_constraint will not
      be reported on for the test set.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataloader: A DataLoader for a data split
      split_name the string naming the data split: {train,dev,test}
    """
    for method in self.reporting_methods:
      if method in self.reporting_method_dict:
        if split_name == 'test' and method not in self.test_reporting_constraint:
          tqdm.write("Reporting method {} not in test set reporting "
              "methods (reporter.py); skipping".format(method))
          continue
        tqdm.write("Reporting {} on split {}".format(method, split_name))
        self.reporting_method_dict[method](prediction_batches
            , dataloader, split_name)
      else:
        tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))

class IndependentLabelReporter(Reporter):
  """
  Class for computing and reporting metrics on
  tasks where each label output of the prediction
  should be compared to its corresponding label,
  and accuracy should be computed by taking the
  percent of correct labels over all outputs
  (not including the pad label).

  This is as opposed to NER, for example, in which
  span-based evaluation is required.

  But works for PoS, dep, maybe even NLI; not sure yet
  """
  yaml_tag = '!IndependentLabelReporter'

  def __init__(self, args, reporting_root, reporting_methods):
    self.args = args
    self.reporting_methods = reporting_methods
    self.reporting_method_dict = {
        'label_accuracy':self.report_label_values,
        'v_entropy':self.report_v_entropy,
        }
    #self.reporting_root = args['reporting']['root']
    self.reporting_root = reporting_root
    self.test_reporting_constraint = {'label_accuracy', 'v_entropy'}


  def report_label_values(self, prediction_batches, dataset, split_name):
    total = 0
    correct = 0
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      if len(prediction_batch.shape) == 3:
        prediction_batch = torch.argmax(prediction_batch, 2)
      else:
        prediction_batch = torch.argmax(prediction_batch, 1)
        label_batch = label_batch.view(label_batch.shape[0])
      agreements = (prediction_batch == label_batch).long()
      filtered_agreements = torch.where(label_batch != 0, agreements,
              torch.zeros_like(agreements))
      total_agreements = torch.sum(filtered_agreements.long())
      total_labels = torch.sum((label_batch != 0).long())
      total += total_labels.cpu().numpy()
      correct += total_agreements.cpu().numpy()

    with open(os.path.join(self.reporting_root, split_name + '.label_acc'), 'w') as fout:
      fout.write(str(float(correct)/  total) + '\n')

  def report_v_entropy(self, prediction_batches, dataset, split_name):
    total_label_count = 0
    neg_logprob_sum = 0
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      batch_label_count = torch.sum((label_batch != 0).long())
      if len(prediction_batch.shape) == 3:
        prediction_batch = torch.softmax(prediction_batch, 2)
        label_batch = label_batch.view(*label_batch.shape, 1)
        prediction_batch = torch.gather(prediction_batch, 2, label_batch)
      else:
        prediction_batch  = torch.softmax(prediction_batch, 1)
        prediction_batch = torch.gather(prediction_batch, 1, label_batch)
        label_batch = label_batch.view(label_batch.shape[0])
        label_batch = label_batch.view(*label_batch.shape, 1)
      batch_neg_logprob_sum = -torch.sum(torch.where((label_batch!=0),
        torch.log2(prediction_batch), torch.zeros_like(prediction_batch)))

      total_label_count += batch_label_count
      neg_logprob_sum += batch_neg_logprob_sum

    with open(os.path.join(self.reporting_root, split_name + '.v_entropy'), 'w') as fout:
      fout.write(str(float(neg_logprob_sum)/float(total_label_count)) + '\n')

class NERReporter(IndependentLabelReporter):
  """ Class for reporting metrics on the Named Entity Recognition task.

  Requires special handling because of entity-level eval and integration
  with the stanza library scorer.
  """
  yaml_tag = '!NERReporter'

  def __init__(self, args, reporting_root, reporting_methods, ner_task):
    """
    Arguments:
      reporting_root: path to which results will be written
      reporting_methods: list of metrics to report
      ner_task: the NERClassificationTask object representing the NER task;
                used to map integer labels to label strings
    """
    self.args = args
    self.reporting_methods = reporting_methods
    self.reporting_method_dict = {
        'label_accuracy':self.report_label_values,
        'v_entropy':self.report_v_entropy,
        'ner_f1':self.report_ner_f1
        }
    #self.reporting_root = args['reporting']['root']
    self.reporting_root = reporting_root
    self.test_reporting_constraint = {'label_accuracy', 'v_entropy', 'ner_f1'}
    self.ner_task = ner_task


  def report_ner_f1(self, prediction_batches, dataset, split_name):
    """
    Reports entity-level NER F1 using the stanza library scorer
    """
    string_predictions = []
    string_labels = []
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      prediction_batch = torch.argmax(prediction_batch, 2)
      for prediction_sentence, label_sentence in zip(prediction_batch, label_batch):
        string_predictions.append(list(filter(lambda x: x != '-', [self.ner_task.category_string_of_label_int(x)
          for x in prediction_sentence])))
        string_labels.append(list(filter(lambda x: x != '-', [self.ner_task.category_string_of_label_int(x)
          for x in label_sentence])))
    precision, recall, f1 = score_by_entity(string_predictions, string_labels)

    with open(os.path.join(self.reporting_root, split_name + '.f1'), 'w') as fout:
      fout.write(str(f1) + '\n')
    with open(os.path.join(self.reporting_root, split_name + '.precision'), 'w') as fout:
      fout.write(str(precision) + '\n')
    with open(os.path.join(self.reporting_root, split_name + '.recall'), 'w') as fout:
      fout.write(str(recall) + '\n')

class WordPairReporter(Reporter):
  """Reporting class for wordpair (distance) tasks"""
  yaml_tag = '!WordPairReporter'

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'image_examples':self.report_image_examples,
        'uuas':self.report_uuas_and_tikz,
        'proj_acc': self.report_proj_nonproj_accuracy,
        'adj_acc': self.report_adj_accuracy,
        """
        TODO: add visulization later 
        """
        # 'write_predictions':self.write_json,
        # 'write_data': self.write_data,
        # 'tsne': self.write_tsne,
        # 'pca': self.write_pca,
        'unproj_tsne': self.write_unprojected_tsne,
    }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}

  def generate_square_dist_matrix(self, length):
    x = np.linspace(-(length-1)/2, (length-1)/2, num=length)
    weighting = np.flip(x).T[:, np.newaxis] + x # black magic
    weighting = np.abs(weighting)
    weighting = torch.FloatTensor(weighting)
    return weighting

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    use_linear = False
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        length = int(length)

        # uncomment this for linear baseline
        if use_linear: prediction = self.generate_square_dist_matrix(length)
        else: prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
        lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr' + ('_linear' if use_linear else '')), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean' + ('_linear' if use_linear else '')), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold distance matrices to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        words = observation.sentence
        fontsize = 5*( 1 + np.sqrt(len(words))/200)
        plt.clf()
        ax = sns.heatmap(label)
        ax.set_title('Gold Parse Distance')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-gold'+str(images_printed)), dpi=300)

        plt.clf()
        ax = sns.heatmap(prediction)
        ax.set_title('Predicted Parse Distance (squared)')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-pred'+str(images_printed)), dpi=300)
        print('Printing', str(images_printed))
        images_printed += 1
        if images_printed == 20:
          return

  def report_uuas_and_tikz(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[uuas,tikz]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        gold_edges = prims_matrix_to_edges(label, words, poses)

        # uncomment this for linear baseline
        # pred_edges = [(i, i+1) for i in range(len(words)-1)]
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        if split_name != 'test' and total_sents < 20:
          self.print_tikz(pred_edges, gold_edges, words, split_name)

        uspan_correct += len(set([tuple(sorted(x)) for x in gold_edges]).intersection(
          set([tuple(sorted(x)) for x in pred_edges])))
        uspan_total += len(gold_edges)
        total_sents += 1
    uuas = uspan_correct / float(uspan_total)
    with open(os.path.join(self.reporting_root, split_name + '.uuas'), 'w') as fout:
      fout.write(str(uuas) + '\n')

  def report_proj_nonproj_accuracy(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    total_nonproj_deps = 0
    total_proj_deps = 0
    correct_nonproj_deps = 0
    correct_proj_deps = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[proj]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        ## calculate which are projective and which are non-projective
        head_indices = [int(x) for x in observation.head_indices]
        is_proj = np.zeros([length])
        def is_projective(dep):
            head = head_indices[dep]-1
            for i in range(min(dep, head) + 1, max(dep, head)):
                curr = head_indices[i]
                while curr != 0 and curr != head + 1:
                    curr = head_indices[curr-1]
                if head != -1 and curr == 0:
                    return False
            return True

        is_proj = [is_projective(dep) for dep in range(length)]

        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        pairs = 0

        out_debug_pairs = []

        for idx, head_idx in enumerate(head_indices):
            if head_idx == 0: continue
            if poses[idx] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-", "PUNCT"]: continue
            head_idx -= 1
            out_debug_pairs.append(tuple(sorted([idx, head_idx])))
            if is_projective(idx):
                total_proj_deps += 1
                pairs += 1
                correct_proj_deps += (tuple(sorted([idx, head_idx])) in pred_edges)
            else:
                total_nonproj_deps += 1
                pairs += 1
                correct_nonproj_deps += (tuple(sorted([idx, head_idx])) in pred_edges)

        out_debug_pairs = sorted(out_debug_pairs)
        assert gold_edges == out_debug_pairs
        verbose = False
        if verbose:
            for word, val in [kv for kv in zip(words, is_proj)]:
                if not val: print("**", end="")
                print(word, end="")
                if not val: print("**", end="")

    correct_deps = correct_proj_deps + correct_nonproj_deps
    total_deps = total_proj_deps + total_nonproj_deps

    with open(os.path.join(self.reporting_root, split_name+'-nonproj.info'), 'w') as fout:
        fout.write(f"Proj: {correct_proj_deps}\n{total_proj_deps}\n{correct_proj_deps/total_proj_deps}\n")
        if total_nonproj_deps != 0:
            fout.write(f"Nonproj: {correct_nonproj_deps}\n{total_nonproj_deps}\n{correct_nonproj_deps/total_nonproj_deps}\n")
        fout.write(f"Total: {correct_deps}\n{total_deps}\n{correct_deps/total_deps}\n")

    with open(os.path.join(self.reporting_root, split_name+'-nonproj.acc'), 'w') as fout:
        if total_nonproj_deps != 0:
            fout.write(f"{correct_nonproj_deps}\n{total_nonproj_deps}\n{correct_nonproj_deps/total_nonproj_deps}")

  def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    ''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
    with open(os.path.join(self.reporting_root, split_name+'.tikz'), 'a') as fout:
      string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """
      string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
      string += "\\end{deptext}" + '\n'
      for i_index, j_index in gold_edges:
        string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      for i_index, j_index in prediction_edges:
        string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      string += '\\end{dependency}\n'
      fout.write('\n\n')
      fout.write(string)

  def report_adj_accuracy(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    total_pre_adj = 0
    correct_pre_adj = 0
    total_post_adj = 0
    correct_post_adj = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[mod]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        head_indices = [int(x)-1 for x in observation.head_indices]
        is_proj = np.zeros([length])


        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        pairs = 0

        for idx, head_idx in enumerate(head_indices):
            if head_idx == -1: continue
            if poses[idx] == 'ADJ' and poses[head_idx] == 'NOUN':
                valid = True
                for i in range(min(idx, head_idx)+1, max(idx, head_idx)): # check that all words in between are adjectives or adverbs
                  print(words[i])
                  if poses[i] not in ('ADJ', 'ADV'):
                     valid = False
                     break
                if valid:
                  if head_idx > idx:
                    total_pre_adj += 1
                    print(correct_pre_adj, pred_edges, idx, head_idx)
                    correct_pre_adj += (tuple(sorted([idx, head_idx])) in pred_edges)
                  else:
                    total_post_adj += 1
                    correct_post_adj += (tuple(sorted([idx, head_idx])) in pred_edges)

    with open(os.path.join(self.reporting_root, split_name+'-adj.info'), 'w') as fout:
        if total_pre_adj:
          fout.write(f"Pre: ({correct_pre_adj}/{total_pre_adj})\t{correct_pre_adj/total_pre_adj}\n")
        if total_post_adj:
          fout.write(f"Post: ({correct_post_adj}/{total_post_adj})\t{correct_post_adj/total_post_adj}\n")

  def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    ''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
    with open(os.path.join(self.reporting_root, split_name+'.tikz'), 'a') as fout:
      string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """
      string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
      string += "\\end{deptext}" + '\n'
      for i_index, j_index in gold_edges:
        string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      for i_index, j_index in prediction_edges:
        string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      string += '\\end{dependency}\n'
      fout.write('\n\n')
      fout.write(string)
