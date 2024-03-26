"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_caption_datasets import VideoCaptionDataset
from lavis.datasets.data_utils import load_video
import pdb

class MSRVTTCapDataset(VideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.video_id_list = list(self.annotation.keys())
        self.video_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, ann['frame_length'], self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        # print(selected_frame_index, video.shape)

        text_input = self.prompt
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "prompt": self.prompt,
            "image_id": ann["video"],
        }
        
    def __len__(self):
        return len(self.video_id_list)

class MSRVTTCapEvalDataset(MSRVTTCapDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val')

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)
        # print(selected_frame_index, video.shape)

        text_input = self.prompt
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "prompt": self.prompt,
            "image_id": ann["video"],
        }
