"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    # job_id = now()

    cfg = Config(parse_args())
    dataset_name = list(cfg.datasets_cfg.keys())[0]
    cfg.datasets_cfg[dataset_name]['num_frames'] = cfg.model_cfg.num_frames

    job_id = '{}/{}_{}/'.format(dataset_name, cfg.model_cfg.arch, cfg.model_cfg.model_type)
    if cfg.run_cfg.prefix:
        job_id += f'{cfg.run_cfg.prefix}/'

    if 'lvu' in dataset_name:
        job_id += '{}/b{}_e{}_lr{}_wd{}_q{}_h{}_f{}_fb{}'.format(
                        cfg.datasets_cfg[dataset_name]['task'], 
                        cfg.run_cfg.batch_size_train * cfg.run_cfg.accum_grad_iters, 
                        cfg.run_cfg.max_epoch, cfg.run_cfg.init_lr, 
                        cfg.run_cfg.weight_decay, cfg.model_cfg.num_query_token,
                        cfg.datasets_cfg[dataset_name]['history'], 
                        cfg.datasets_cfg[dataset_name]['num_frames'], 
                        cfg.model_cfg.memory_bank_length,
                    )
    elif 'coin' in dataset_name or 'breakfast' in dataset_name:
        job_id += 'b{}_e{}_lr{}_wd{}_q{}_f{}_fb{}'.format(
                        cfg.run_cfg.batch_size_train * cfg.run_cfg.accum_grad_iters, 
                        cfg.run_cfg.max_epoch, cfg.run_cfg.init_lr, 
                        cfg.run_cfg.weight_decay, cfg.model_cfg.num_query_token,
                        cfg.datasets_cfg[dataset_name]['num_frames'], 
                        cfg.model_cfg.memory_bank_length,
                    )
    elif 'msvd' in dataset_name or 'msrvtt' in dataset_name or 'activitynet' in dataset_name or 'youcook' in dataset_name:
        job_id += 'b{}_e{}_lr{}_wd{}_q{}_f{}_fb{}'.format(
                        cfg.run_cfg.batch_size_train * cfg.run_cfg.accum_grad_iters, 
                        cfg.run_cfg.max_epoch, cfg.run_cfg.init_lr, 
                        cfg.run_cfg.weight_decay, cfg.model_cfg.num_query_token,
                        cfg.datasets_cfg[dataset_name]['num_frames'], 
                        cfg.model_cfg.memory_bank_length,
                    )

    cfg.model_cfg.arch += '_malmm'
    if cfg.model_cfg.freeze_vit:
        job_id += '_freezevit'

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    cfg.run_cfg.log_dir = os.path.join('lavis', cfg.run_cfg.output_dir, job_id)
    setup_logger(output_dir=os.path.join('lavis', cfg.run_cfg.output_dir, job_id))

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
