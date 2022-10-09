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

"""Main function to train and eval vatt models."""

import pprint

from absl import app
from absl import flags
from absl import logging

from vatt.configs import factory as config_factory
from vatt.experiments import finetune
from vatt.experiments import pretrain

flags.DEFINE_string('task', 'PRETRAIN', 'PRETRAIN or FINETUNE.')
flags.DEFINE_string('mode', 'train', 'train or eval.')
flags.DEFINE_string('model_dir', None, 'Default path for the experiment.')
flags.DEFINE_string('model_arch', 'Tx_FAC', 'Arch of the model.')
flags.DEFINE_string('override_checkpoint', None,
                    ('Path to a checkpoint for initialization. '
                     'If this is passed, the model is initialized from this '
                     'checkpoint, even if there is a valid latest checkpoint '
                     'inside the model_dir.'))
flags.DEFINE_string('config_file', None,
                    ('Path to a YAML config file containing the dictionary of '
                     'parameters to override the original params defined '
                     'under configs/'))
flags.DEFINE_string('params_override', None,
                    'A safe_dumped str of a dictionary of parameters')
flags.DEFINE_string('strategy_type', 'tpu', 'Type of the distribution strategy')
flags.DEFINE_string('tpu', None, 'Address of the TPU device')


FLAGS = flags.FLAGS


def get_params():
  """Constructs the configuration of the experiment."""

  task = FLAGS.task
  model_arch = FLAGS.model_arch
  params = config_factory.build_experiment_configs(
      task=task,
      model_arch=model_arch,
      )

  if FLAGS.config_file:
    params.override_from_file(FLAGS.config_file)

  if FLAGS.params_override:
    params.override_from_str(FLAGS.params_override)

  params.override({
      'mode': FLAGS.mode,
      'model_dir': FLAGS.model_dir,
      'checkpoint_path': FLAGS.override_checkpoint,
      'strategy_config': {'tpu': FLAGS.tpu,
                          'distribution_strategy': FLAGS.strategy_type},
  })

  return params


def main(argv):
  del argv  # Unused.
#python vatt.main --task=pretrain --mode=train --model_dir=PATH/TO/RUN --model_arch=tx_fac --strategy_type=mirrored
  params = get_params()
  logging.info('Model Parameters: %s', pprint.pformat(params.as_dict()))
  """
  'eval': {'input': {'audio_mixup': False,
                    'audio_stride': 1,
                    'batch_size': 8,
                    'color_augment': True,
                    'crop_resize_style': 'VGG',
                    'frame_size': 224,
                    'has_data': True,
                    'linearize_vision': True,
                    'max_area_ratio': 1.0,
                    'max_aspect_ratio': 2.0,
                    'max_num_words': 16,
                    'mel_bins': 80,
                    'min_area_ratio': 0.08,
                    'min_aspect_ratio': 0.5,
                    'min_resize': 224,
                    'mixup_alpha': 10,
                    'mixup_beta': 1,
                    'multi_crop': False,
                    'name': ('esc50', 'hmdb51', 'ucf101', 'youcook2', 'msrvtt'),
                    'num_augmentation': 1,
                    'num_examples': 4096,
                    'num_frames': 32,
                    'num_windows_test': 4,
                    'raw_audio': True,
                    'scale_jitter': True,
                    'space_to_depth': False,
                    'split': 'test',
                    'stft_length': 0.04267,
                    'stft_step': 0.02134,
                    'text_tokenizer': 'WordTokenizer',
                    'video_stride': 2,
                    'zero_centering_image': True}},
 'mode': 'train',
 'model_config': {'backbone_config': {'audio_backbone': 'wat_base',
                                      'audio_model_kwargs': {},
                                      'name': 'backbone_stack',
                                      'text_backbone': 't5_small',
                                      'text_model_kwargs': {},
                                      'video_backbone': 'vit_medium',
                                      'video_model_kwargs': {}},
                  'head_config': {'bridge': ({'aud_to_vid_txt_kwargs': {'d_model': 512,
                                                                        'modality': 'audio',
                                                                        'name': 'audio_mlp_module'},
                                              'bn_config': {'epsilon': 1e-05,
                                                            'momentum': 0.9,
                                                            'name': 'batch_norm',
                                                            'scale': True},
                                              'name': 'mlp_fac',
                                              'txt_to_vid_aud_kwargs': {'d_model': 256,
                                                                        'modality': 'text',
                                                                        'name': 'text_mlp_module'},
                                              'use_xreplica_bn': True,
                                              'vid_to_aud_txt_kwargs': {'d_model': 512,
                                                                        'modality': 'video',
                                                                        'name': 'video_mlp_module'}},)},
                  'loss_config': {'bridge': ({'aud_txt_weight': 0.0,
                                              'loss_weight': 1.0,
                                              'name': 'asymmetric_nce',
                                              'temperature': 0.07,
                                              'vid_aud_weight': 1.0,
                                              'vid_txt_weight': 1.0},)},
                  'model_name': 'tx_mlp_fac'},
 'model_dir': 'PATH/TO/RUN',
 'strategy_config': {'distribution_strategy': 'mirrored', 'tpu': None},
 'task': 'Pretrain',
 'train': {'gradient_clip_norm': 0.0,
           'gradient_clip_norm_cls': None,
           'input': {'audio_mixup': False,
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
                     'zero_centering_image': True},
           'iterations_per_loop': 50,
           'max_checkpoints': 50,
           'optimizer': {'beta_1': 0.9,
                         'beta_2': 0.999,
                         'epsilon': 1e-07,
                         'learning_rate': {'learning_rate_base': 0.0001,
                                           'learning_rate_levels': (0.0001,
                                                                    5e-05),
                                           'learning_rate_steps': (5000,
                                                                   500000),
                                           'total_steps': 500000,
                                           'warmup_learning_rate': 0.0,
                                           'warmup_steps': 5000},
                         'name': 'Adam'},
           'save_checkpoint_freq': 10000}}
  """
  if params.task.lower() == 'pretrain':
    executor = pretrain.get_executor(params=params)

  elif params.task.lower() == 'finetune':
    executor = finetune.get_executor(params=params)

  else:
    raise ValueError('Task not found: %s.' % params.task)

  return executor.run(mode=params.mode)


if __name__ == '__main__':
  app.run(main)
