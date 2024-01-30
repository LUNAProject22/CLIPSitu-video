# %%
import torch
from PIL import Image
from collections import OrderedDict, defaultdict
import pickle, pickletools, gzip
import os
import numpy as np
import json 
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from transformers import XCLIPModel, XCLIPProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)
model.to(device)

from decord import VideoReader, cpu
import numpy as np

np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


dataset_dir = '/data/dataset/VidSitu/data/'
anno_dir = os.path.join(dataset_dir, 'vidsitu_annotations/split_files/')
video_dir = os.path.join(dataset_dir, 'vsitu_video_trimmed_dir/')


class VidSituDataset(object):
    
    def __init__(self, json_file):
        self.annotations = []
                
        self.annotations = self._read_data(json_file)
        
        self.embeddings = {}
        for i in tqdm(range(len(self.annotations))):
            encoding = self.getitem(i)
            if encoding['video_emb'] is not None:
                self.embeddings[encoding['video_id']] = encoding
            # if (i+1)%100 == 0:
            #     break

    def __len__(self):
        return len(self.annotations)
    
       
    def _read_data(self, json_file):
        annotations = []
        json_path = os.path.join(anno_dir, json_file)
        # print(json_path)
        with open(json_path, "r") as f:
            video_list = json.load(f)
            count = 0
            # print(video_list)
            for vidname in video_list:
                vidpath = os.path.join(video_dir, vidname)
                
                annotation = {}
                annotation['video'] = vidpath + '.mp4'
                annotations.append(annotation)   
        return annotations
    
    def getitem(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        # print(annotation)
        vid_path = annotation['video']
        

        # sample 8 frames
        
        encoding ={}
        encoding["video_id"] = annotation['video'].split('/')[-1]
        try:
            vr = VideoReader(vid_path, num_threads=1, ctx=cpu(0))
            vr.seek(0)
            indices = sample_frame_indices(clip_len=20, frame_sample_rate=2, seg_len=len(vr)) # processing at 2 fps
            video = vr.get_batch(indices).asnumpy()
            
            video_input = processor(text=["."], videos=list(video), return_tensors="pt", padding=True)
            outputs =  model.forward(**video_input)
           
            encoding["video_emb"] = outputs.vision_model_output['last_hidden_state'].detach().cpu()     
        except:
            print('not found', annotation['video'])
            encoding["video_emb"] = None
            return encoding
        
        return encoding

json_list = [
            #  'vseg_split_testevrel_lb.json',
            #  'vseg_split_testsrl_lb.json',
            #  'vseg_split_testvb_lb.json',
             'vseg_split_train_lb.json',
             'vseg_split_valid_lb.json'
            ]

feats = []
for json_file in json_list:
    dataset = VidSituDataset(json_file)
    feats.append(dataset.embeddings)

    
save_folder = '../vidsitu_xclip_xtf_features' #'/data/usrdata/roy/imsitu_clip_xtf_features'
with open(os.path.join(save_folder, 'xclip_b32.pkl'), 'wb') as f:
    # pickled = pickle.dumps(overall_feat, protocol=pickle.HIGHEST_PROTOCOL)
    # optimized_pickle = pickletools.optimize(pickled)
    # f.write(optimized_pickle)
    for feat in feats:
        pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)