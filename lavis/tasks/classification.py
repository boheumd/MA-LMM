"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import numpy as np

import torch.distributed as dist
from lavis.common.dist_utils import main_process, get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample

import pdb

@registry.register_task("classification")
class ClassificationTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, 
                 report_metric=True, verb_only=False, noun_only=False,
                 dataset_name=None, log_dir=None):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
        
        self.verb_only = verb_only
        self.noun_only = noun_only
        self.dataset_name = dataset_name
        self.log_dir = log_dir

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        log_dir = run_cfg.log_dir

        report_metric = run_cfg.get("report_metric", True)
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            dataset_name=list(cfg.datasets_cfg.keys())[0],
            log_dir=log_dir,
        )

    def valid_step(self, model, samples):
        results = []
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            num_captions=self.num_beams,
        )

        img_ids = samples["image_id"]
        batch_size = len(img_ids)
        for i, img_id in enumerate(img_ids):
            caption_list = captions[i * self.num_beams : (i + 1) * self.num_beams]
            results.append({"caption": caption_list, "image_id": img_id})
        return results

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        results = []
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()
        return results

    def after_evaluation(self, val_result, split_name, epoch, dataset, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics_cls(
                eval_result_file=eval_result_file, split_name=split_name, dataset=dataset
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics_cls(self, eval_result_file, split_name, dataset):
        gt_dict = dataset.annotation

        with open(eval_result_file, 'r') as fp:
            prediction_list = json.load(fp)
        
        dataset_size = len(prediction_list)

        acc_sum = 0
        image_id_list = []
        
        match_video_list = []
        for prediction in prediction_list:
            image_id = prediction['image_id']
            caption_list = prediction['caption']
            image_id_list.append(image_id)
            label = gt_dict[image_id]['label']
            
            match_video = [1 if caption == label else 0 for caption in caption_list]
            match_video_list.append(match_video)
        match = np.array(match_video_list)
        
        top_1 = match[:, :1].max(1).mean() * 100
        top_5 = match[:, :5].max(1).mean() * 100

        result = {
            'top1': top_1, 'top5': top_5,
        }

        print(f"top1: {top_1:.2f} top5: {top_5:.2f}\n")
        result['agg_metrics'] = result['top1']
        return result

