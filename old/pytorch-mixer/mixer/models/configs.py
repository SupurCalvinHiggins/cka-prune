# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_mixer_s8_config():
    """Returns Mixer-S/8 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_8'
    config.patch_size = 8
    config.hidden_dim = 512
    config.num_blocks = 8
    config.tokens_mlp_dim = 256
    config.channels_mlp_dim = 2048
    return config


def get_mixer_s16_config():
    """Returns Mixer-S/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_16'
    config.patch_size = 16
    config.hidden_dim = 512
    config.num_blocks = 8
    config.tokens_mlp_dim = 256
    config.channels_mlp_dim = 2048
    return config


def get_mixer_s32_config():
    """Returns Mixer-S/32 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_32'
    config.patch_size = 32
    config.hidden_dim = 512
    config.num_blocks = 8
    config.tokens_mlp_dim = 256
    config.channels_mlp_dim = 2048
    return config


def get_mixer_b16_config():
    """Returns Mixer-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patch_size = 16
    config.hidden_dim = 768
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    return config


def get_mixer_b32_config():
    """Returns Mixer-B/32 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_32'
    config.patch_size = 32
    config.hidden_dim = 768
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    return config


def get_mixer_l16_config():
    """Returns Mixer-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-L_16'
    config.patch_size = 16
    config.hidden_dim = 1024
    config.num_blocks = 24
    config.tokens_mlp_dim = 512
    config.channels_mlp_dim = 4096
    return config