import os.path as op
import random
from typing import List
from datasets.cap2img import GPTDIFFBLIP_PEDES

from utils.iotools import read_json
from .bases import BaseDataset

import os
import json
from prettytable import PrettyTable
import collections
import numpy as np

class LuPerson_PEDES(BaseDataset):
    dataset_dir = 'LUPerson_images'
    trainSet = ['train_15w_part1','train_15w_part2','train_15w_part3','train_15w_part4']
    testSet = []
    def __init__(self, root='', verbose=True):
        super(LuPerson_PEDES, self).__init__()
        self.dataset_dir = '/export/home/tanwentao1/data/LuPerson-T'
        self.image_dir = op.join(self.dataset_dir, 'train')
        self.caption_dir = self.dataset_dir
        self.train_img_paths = []
        self.train_cap_paths = []

        self.test_img_paths = []
        self.test_cap_paths = []

        for filename in os.listdir(self.image_dir): # part1234
            image_path = os.path.join(self.image_dir, filename)
            if filename.endswith('.jpg'):
                self.train_img_paths.append(image_path)
        for filename in os.listdir(self.caption_dir):
            caption_path = os.path.join(self.caption_dir, filename)
            if filename.endswith('.json'):
                self.train_cap_paths.append(caption_path)
        
        train_cap_dict = self._merged_multi_json_file(self.train_cap_paths)
        test_cap_dict = self._merged_json_file(self.test_cap_paths)

        self.train, self.train_id_container, self.part_dataset, num_caption,self.fpath2part_cap,self.fpaht2sim = self._get_dataset(self.train_img_paths, train_cap_dict)
        self.test = self._get_test_dataset(self.test_img_paths, test_cap_dict)
        # syn_dataset = GPTDIFFBLIP_PEDES(root=root)
        
        # self.train = self.train + syn_dataset.train
        
        self.logger.info("=> LuPerson-15w Images and Captions are loaded")
        self.logger.info("LuPerson-15w Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', len(set(self.train_id_container)),len(self.train), num_caption])
        table.add_row(['test', len(self.test["image_pids"]),len(self.test["image_pids"]), len(self.test["image_pids"])])
        self.logger.info('\n' + str(table))
        

    def _merged_json_file(self, json_path_list):
        merged_dict = {}

        # 逐个读取JSON文件并合并到字典中
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                merged_dict.update(data)

        import json
        import random

        # 读取JSON文件
        with open('your_large_json_file.json', 'r') as file:
            data = json.load(file)

        # 确保键的数量至少有10万个
        if len(merged_dict) < 100000:
            print("文件中的键数量少于10万。")
        else:
            # 从所有键中随机选取10万个
            random_keys = random.sample(list(merged_dict.keys()), 100000)
            
            with open('random_100k_keys_json_file.json', 'w') as outfile:
                json.dump(random_keys, outfile)
            
            print("已经随机选取10万个键并保存到新的JSON文件中。")
        return merged_dict
    
    def _merged_multi_json_file(self, json_path_list):
        merged_dict = collections.defaultdict(list)
        json_path_list = [
                          "./caption/Lup_qwen.json",
                          "./caption/Lup_shikra.json",
                          ]
        for temp_file in os.listdir('./caption/c0'):
            json_path_list.append(os.path.join('./caption/c0',temp_file))
        for temp_file in os.listdir('./caption/shikra'):
            json_path_list.append(os.path.join('./caption/shikra',temp_file))
        for file_path in json_path_list:
            # if 'qwen_caption_part0' not in file_path: continue
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                print(file_path, len(data))
                for k,v in data.items():
                    img_name = k.split('/')[-1]
                    # for v_i in v:
                    merged_dict[img_name].append(v[-1])
        return merged_dict

    def _get_test_dataset(self, test_img_paths, cap_dict):
        dataset = {}
        img_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        for i in range(len(test_img_paths)):
            pid = i
            img_path = test_img_paths[i]
            img_paths.append(img_path)
            image_pids.append(pid)
            path2cap = '/'.join(img_path.split('/')[-1])
            caption = cap_dict[path2cap][0]
            captions.append(caption)
            caption_pids.append(pid)
        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions
        }
        return dataset
    
    def _get_dataset(self, img_paths, cap_dict):

        safe_dict = collections.defaultdict(list)
        with open('./caption/Lup_shikra.json', 'r') as json_file:
            data = json.load(json_file)
            for k,v in data.items():
                img_name = k.split('/')[-1]
                safe_dict[img_name].append(v[-1])
        
        with open('./caption/Lup_qwen.json', 'r') as json_file:
            data = json.load(json_file)
            for k,v in data.items():
                img_name = k.split('/')[-1]
                safe_dict[img_name].append(v[-1])
        pid_container = set()
        img_paths = sorted(img_paths)

        dataset = []
        part_dataset = []
        idx_count = 0
        pid_count = 0
        num_caption = 0

        fpath2part_cap = {}
        fpaht2sim = {}
        for i in range(len(img_paths)):
            if pid_count == 100000:break
            img_path = img_paths[i]
            
            path2cap = img_path.split('/')[-1]
            caption = cap_dict[path2cap]
            if len(caption) != 4:
                continue
            fpath2part_cap[img_path] = {}
            fpaht2sim[img_path] = {}
            pid = pid_count
            image_id = idx_count
            pid_container.add(pid)
            for cap in caption:
                if 'description]' in cap or '<' in cap: 
                    cap = random.choice(safe_dict[path2cap])
                part2sim = 77 * [1- 0.15]
                part2sim = np.array(part2sim)
                dataset.append([pid,idx_count,img_path, cap, part2sim])
                num_caption += 1
                idx_count += 1
            pid_count += 1
        assert idx_count == len(dataset)
        return dataset, pid_container, part_dataset,num_caption,fpath2part_cap,fpaht2sim