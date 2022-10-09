# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base factories which wraps all loading/decoding/processing modules."""

import abc
import os
import pprint
from typing import Any, Union, Optional, List, Mapping

from absl import logging
from dmvr import builders
from dmvr import sources
from dmvr import video_dataset as dmvr_base
import tensorflow as tf

from vatt.data import processing


# ----------------------------------------------------------------------
# --------------------------- Base factories ---------------------------
# ----------------------------------------------------------------------

RAWFORMATS = {
    "tf_example": builders.ExampleParserBuilder,
    "tf_sequence_example": builders.SequenceExampleParserBuilder,
}
Config = Any


def get_source(source_type):
  if source_type.lower() == "tfrecord":
    return sources.TFRecordsSource
  else:
    raise NotImplementedError

"""DMVR（decode MV refinement）解码端MV细化。

顾名思义，就是将MV在解码端做一个细化操作，那么相应的，就可以不在编码端做过多的搜索细化等操作。
好处是可以减小传输的码率，缩短编码时间。为了保证质量，将细化的任务放在解码端，
虽然增加了一部分解码时间，但在一定条件下，可以将增加的解码时间减小到最少。
"""
class BaseDMVRFactory(dmvr_base.BaseVideoDatasetFactory, abc.ABC):
  """Factory for datasets from a filesystem directly."""

  def __init__(
      self,
      base_dir,
      table,
      source = "tfrecord",
      raw_data_format = "tf_sequence_example"):
    """Initializes the `BaseDMVRFactory`.

    Args:
      base_dir: The path to the base directory of the dataset, where the
        SSTables can be found.
      table: The SSTable to be read. Available tables must be provided via
        `tables` method.
      source: The method which the tables are stored.
      raw_data_format: Format of serialized raw data. See `builders.RawFormat`.

    Raises:
      ValueError: Table name does not exist.
    """
    tables_dict = self.tables()
    if table not in tables_dict:
      raise ValueError(f"Invalid table \'{table}\'. "
                       f"The available tables are: {tables_dict.keys()}.")
    table_relative_path = tables_dict[table]
    if isinstance(table_relative_path, list):
      shards = [os.path.join(base_dir, x) for x in table_relative_path]
    else:
      table_path = os.path.join(base_dir, table_relative_path)
      shards = processing.get_sharded_files(table_path=table_path)

    self.source = get_source(source)
    parser_builder_class = RAWFORMATS[raw_data_format]
    super().__init__(shards=shards,
                     parser_builder_class=parser_builder_class,
                     source=self.source())

  @abc.abstractmethod
  def tables(self):
    """Returns a dictionary from table name to relative path."""

# ----------------------------------------------------------------------
# ------------------------- Base Data Loaders --------------------------
# ----------------------------------------------------------------------


class BaseLoader(object):
  """A generic data loader based on DMVR with multi-dataset support."""

  def __init__(self,
               dmvr_factory,
               params,
               postprocess_fns = None,
               num_epochs = 1,
               mode = "",
               name = ""):
    self.dmvr_factory = [
        dmvr_factory
    ] if not isinstance(dmvr_factory, list) else dmvr_factory
    self.batch_size = params.batch_size
    self.num_epochs = num_epochs
    self.postprocess_fns = postprocess_fns
    self.shuffle = mode == "train"
    self.mode = mode
    self.name = name
    """
    'audio_noise': 0.01,
 'audio_stride': 1,
 'batch_size': 8,
 'color_augment': True,
 'crop_resize_style': 'VGG',
 'frame_size': 224,
 'has_data': True,
 'linearize_vision': True,
 'max_area_ratio': 1.0,
 'max_aspect_ratio': 2.0,
 'max_context_sentences': 4,
 'max_num_words': 16,
 'mel_bins': 80,
 'min_area_ratio': 0.08,
 'min_aspect_ratio': 0.5,
 'min_resize': 224,
 'mixup_alpha': 10,
 'mixup_beta': 2,
 'name': 'howto100m+audioset',
 'num_examples': -1,
 'num_frames': 32,
 'raw_audio': True,
 'scale_jitter': True,
 'space_to_depth': False,
 'split': 'train',
 'stft_length': 0.04267,
 'stft_step': 0.02134,
 'text_tokenizer': 'WordTokenizer',
 'video_stride': 1,
 'zero_centering_image': True}
    """
    # Tune dmvr_factory for large-scale runs
    for factory in self.dmvr_factory:
      factory.tune(
          cycle_length=32,
          num_parallel_calls_interleave=32,
          block_length=1,
          )

  def dataset_fn(self, input_context):
    """Function to construct the data graph and return tf.data.Dataset."""

    # If under tf.distribute.Strategy, shard data to num_replica splits
    if input_context:
      per_replica_batch_size = input_context.get_per_replica_batch_size(
          self.batch_size
          )
      logging.info(
          ("Global batch_size = %s distributed to per-replica batch_size = %s"),
          self.batch_size,
          per_replica_batch_size,
          )
    else:
      per_replica_batch_size = self.batch_size


# Initialize tokenizer, if any
    for factory in self.dmvr_factory:
      if hasattr(factory, "tokenizer"):
        factory.tokenizer.initialize()

    datasets = []
    for factory in self.dmvr_factory:
      datasets.append(
          factory.make_dataset(
              shuffle=self.shuffle,
              num_epochs=self.num_epochs,
              batch_size=8,
              padded_batch=False,
              drop_remainder=True,
              keep_key=False,
              override_preprocess_fn=None,
              )
          )

    if len(datasets) > 1:
      datasets = [ds.unbatch() for ds in datasets]
      combined_ds = tf.data.experimental.sample_from_datasets(datasets)
      dataset = combined_ds.batch(batch_size=per_replica_batch_size,
                                  drop_remainder=True)
    else:
      dataset = datasets[0]

    if self.postprocess_fns:
      # NOTE: Always executed after dataset.batch().
      for p_fn in self.postprocess_fns:
        dataset = dataset.map(p_fn, num_parallel_calls=2)

    return dataset

  def __call__(self,
               input_context = None):
    """Call the dataset_fn with or without tf.data service."""

    dataset = self.dataset_fn(input_context)

    # Perform a final prefetch on local hosts
    dataset = dataset.prefetch(16)

    return dataset
