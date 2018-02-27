#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim
FLAGS_checkpoint_exclude_scopes=None
FLAGS_checkpoint_include_scopes=None #'resnet_v1_50'
def get_var_list_to_restore_by_name(var_name=None):
  """Choose which vars to restore, ignore vars by setting --checkpoint_exclude_scopes """

  variables_to_restore = []
  if FLAGS_checkpoint_exclude_scopes is not None:
    exclusions = [scope.strip()
                  for scope in FLAGS_checkpoint_exclude_scopes.split(',')]

    # build restore list
    for var in tf.model_variables():
      for exclusion in exclusions:
        if var.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)
  else:
    variables_to_restore = tf.model_variables()

  variables_to_restore_final = []
  if var_name is not None:
      includes = [
              scope.strip()
              for scope in var_name.split(',')
              ]
      for var in variables_to_restore:
          for include in includes:
              if var.name.startswith(include):
                  variables_to_restore_final.append(var)
                  break
  else:
      variables_to_restore_final = variables_to_restore

  return variables_to_restore_final
def get_var_list_to_restore():
  """Choose which vars to restore, ignore vars by setting --checkpoint_exclude_scopes """

  variables_to_restore = []
  if FLAGS_checkpoint_exclude_scopes is not None:
    exclusions = [scope.strip()
                  for scope in FLAGS_checkpoint_exclude_scopes.split(',')]

    # build restore list
    for var in tf.model_variables():
      for exclusion in exclusions:
        if var.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)
  else:
    variables_to_restore = tf.model_variables()

  variables_to_restore_final = []
  if FLAGS_checkpoint_include_scopes is not None:
      includes = [
              scope.strip()
              for scope in FLAGS_checkpoint_include_scopes.split(',')
              ]
      for var in variables_to_restore:
          for include in includes:
              if var.name.startswith(include):
                  variables_to_restore_final.append(var)
                  break
  else:
      variables_to_restore_final = variables_to_restore

  return variables_to_restore_final
