import os
import ffmpeg
import threading
import shutil
from tqdm import tqdm

import pdb

dataset = 'breakfast'
src_base_dir = f'{dataset}/videos'
dst_base_dir = f'{dataset}/frames'
fps = 10

def split_list(l, n):
    """Yield successive n-sized chunks from l."""
    length = len(l)
    chunk_size = length // n + 1
    for i in range(0, length, chunk_size):
        yield l[i:i + chunk_size]

def extract_frames(video_name):
    video_id = video_name.split('.')[0]
    os.makedirs("{}/{}".format(dst_base_dir, video_id), exist_ok=True)
    cmd = 'ffmpeg -i \"{}/{}\" -vf scale=-1:256 -pix_fmt yuvj422p -q:v 1 -r {} -y \"{}/{}/frame%06d.jpg\"'.format(
        src_base_dir, video_name, fps, dst_base_dir, video_id, fps)
    os.system(cmd)

def target(full_id_list):
    for video_id in tqdm(full_id_list):
        extract_frames(video_id)

if __name__ == '__main__':
    full_name_list = []
    for video_name in os.listdir(src_base_dir):
        if not os.path.exists("{}/{}".format(src_base_dir, video_name)):
            continue
        full_name_list.append(video_name)

    full_name_list.sort()
    NUM_THREADS = 4
    splits = list(split_list(full_name_list, NUM_THREADS))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()