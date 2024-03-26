"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import lavis.common.utils as utils
import os

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.datasets.msvd_vqa_datasets import MSVDVQADataset, MSVDVQAEvalDataset
from lavis.datasets.datasets.msrvtt_vqa_datasets import MSRVTTVQADataset, MSRVTTVQAEvalDataset
from lavis.datasets.datasets.activitynet_vqa_datasets import ActivityNetVQADataset, ActivityNetVQAEvalDataset
class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets


@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    train_dataset_cls = MSRVTTVQADataset
    eval_dataset_cls = MSRVTTVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }
    # from BaseDatasetBuilder
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        ans2label = ann_info.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)
            
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            if is_train:
                prompt = self.config.get("text_processor").get('train').get('prompt')
            else:
                prompt = self.config.get("text_processor").get('eval').get('prompt')

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
            )
            datasets[split]._build_class_labels(ans2label)

        return datasets

@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    train_dataset_cls = MSVDVQADataset
    eval_dataset_cls = MSVDVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }

    # from BaseDatasetBuilder
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        ans2label = ann_info.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)
            
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            if is_train:
                prompt = self.config.get("text_processor").get('train').get('prompt')
            else:
                prompt = self.config.get("text_processor").get('eval').get('prompt')

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
            )
            datasets[split]._build_class_labels(ans2label)

        return datasets

@registry.register_builder("activitynet_qa")
class ActivityNetQABuilder(VideoQABuilder):
    train_dataset_cls = ActivityNetVQADataset
    eval_dataset_cls = ActivityNetVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/activitynet/defaults_qa.yaml",
    }
    # from BaseDatasetBuilder
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            if is_train:
                prompt = self.config.get("text_processor").get('train').get('prompt')
            else:
                prompt = self.config.get("text_processor").get('eval').get('prompt')

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
            )

        return datasets

