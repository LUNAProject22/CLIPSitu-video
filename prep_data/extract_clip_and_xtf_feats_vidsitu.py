import torch
import json
import os
import pickle
import random
import time
import argparse

from datetime import datetime
from torch import nn

from torch.nn.functional import softmax
from torch.utils.data import Dataset
import clip
import av
import numpy as np
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel


torch.manual_seed(42)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

clip_model_version = 'clip-vit-base-patch16' #'clip-vit-large-patch14-336' # clip-vit-base-patch32
clip_model_version_code = 'ViT-L/14@336px' # 'ViT-B/32'

device = "cuda" if torch.cuda.is_available() else "cpu"
#clip_model, clip_preprocess = clip.load(clip_model_version_code, device=device)

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/"+clip_model_version)
clip_model = CLIPModel.from_pretrained("openai/"+clip_model_version)
clip_text_model = CLIPTextModel.from_pretrained("openai/"+clip_model_version)
image_processor = AutoProcessor.from_pretrained("openai/"+clip_model_version)
clip_model.to(device)

noun2id = pickle.load(open('noun2id.pkl','rb'))

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_vid_embeddings(vid_path, event): # clip embeddings for every 2 sec video
    
    try:
        container = av.open(vid_path)
    except:
        print('video not found')
        return None
    try:
        indices = sample_frame_indices(clip_len=40, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
    except:
        print('low>high')
        return None
    video = read_video_pyav(container, indices)
    
    
    
    # divide video into 5 segments of 2 seconds each 
    chunk_size = int(video.shape[0]/5)
    total_frames = np.arange(video.shape[0])
    frame_ranges = [total_frames[i:i + chunk_size] for i in range(0, video.shape[0], chunk_size)]
    
    # choose the correct segment 0 - 4
    frame_range = frame_ranges[event]
    xtf_vid_features = []
    vid_features = []
    for i in frame_range: # 8 frames
        image = Image.fromarray(video[i,:])
        image_input = image_processor(images=image, return_tensors="pt").to(device)
        #image = image_processor(Image.fromarray(video[i,:])).unsqueeze(0).to(device)
        #vid_features.append(clip_model.encode_image(image_input).detach().cpu())
        outputs =  clip_model.vision_model.forward(**image_input)
        image_emb = clip_model.visual_projection(outputs['pooler_output']).detach().cpu()
        xtf_image_emb = clip_model.visual_projection(outputs['last_hidden_state']).detach().cpu() 
        vid_features.append(image_emb)  
        xtf_vid_features.append(xtf_image_emb)
    with torch.no_grad():
        vid_feat = torch.mean(torch.stack(vid_features).squeeze(), dim=0)
        xtf_vid_feat = torch.mean(torch.stack(xtf_vid_features).squeeze(), dim=0)
    # TODO
    return vid_feat, xtf_vid_feat

def get_verb_embeddings(verb):
    #verb_tok = clip.tokenize(verb).to(device)
    #verb_feat = clip_model.encode_text(verb_tok).detach().cpu()

    verb_tok = clip_tokenizer(verb, padding=True, return_tensors="pt")
    verb_feat = clip_text_model(**verb_tok)['pooler_output'].detach().cpu()
    
    return verb_feat

def get_role_embeddings(role_dict):
    
    roles_feat = []
    for role in role_dict.keys():
        #role_tok = clip.tokenize(role).to(device)
        #role_feat = clip_model.encode_text(role_tok).detach().cpu()
        role_tok = clip_tokenizer(role, padding=True, return_tensors="pt")
        role_feat =  clip_text_model(**role_tok)['pooler_output'].detach().cpu()
        roles_feat.append(role_feat)
    
    return torch.stack(roles_feat).squeeze()

def get_label_embeddings(role_dict):
    
    labels_feat = []
    for role in role_dict.keys():
        #label_tok = clip.tokenize(role_dict[role]).to(device)
        #label_feat = clip_model.encode_text(label_tok).detach().cpu()
        label_tok = clip_tokenizer(role_dict[role], padding=True, return_tensors="pt")
        label_feat = clip_text_model(**label_tok)['pooler_output'].detach().cpu()
        labels_feat.append(label_feat)
        
    return torch.stack(labels_feat).squeeze()

def get_label_ids(role_dict):
    labels_id = []
    for role in role_dict.keys():
        id = noun2id[role_dict[role]]
        labels_id.append(id)
    
    return labels_id

class VidSituDataset(object):
    def __init__(self, split='train'): # split is train or valid
        #root_dir = "/data/dataset/VidSitu/data/"
        root_dir = "/home/dhruv/Projects/VidSitu/vidsitu_data/"
        split_dir = os.path.join(root_dir, "vidsitu_annotations/split_files")
        split_name = os.path.join(split_dir, 'vseg_split_' + split + '_lb.json')
        self.vids = json.load(open(split_name,'r'))
        
        self.vid_trimmed_dir = os.path.join(root_dir, "vsitu_video_trimmed_dir")
        
        anno_dir = os.path.join(root_dir, "vidsitu_annotations/vseg_ann_files")
        anno_name = os.path.join(anno_dir, 'vsann_' + split + '_lb.json')
        annos = json.load(open(anno_name,'r'))
        
        self.annotations = []
        for anno in annos:
            
            for event in anno.keys(): # should return Ev1,...,Ev5
                annotation = {}
                event_info = anno[event]
                vid_name = event_info['vid_seg_int']
                # if vid_name == 'v_JEcfknHqMyc_seg_80_90':
                    # sanity check - check if video exists
                if os.path.exists(os.path.join(self.vid_trimmed_dir, vid_name+'.mp4')):
                    annotation['vid_path'] = os.path.join(self.vid_trimmed_dir, vid_name+'.mp4')
                    annotation['verb'] = event_info['Verb'].split()[0]
                    role_dict = OrderedDict()
                    args = event_info['Args']
                    # Role should be Arg0, Arg1, Arg2, ArgM (location) or Scene of the Event
                    for key in args.keys():
                        role = key.strip(')').split('(')[-1] 
                        arg_name = key.split()[0]
                        if arg_name in  ['Arg0', 'Arg1', 'Arg2', 'ArgM']:
                            role_dict[role] = args[key]
                        if arg_name == 'Scene':
                            role_dict[arg_name] = args[key]
                    annotation['role_dict'] = role_dict
                    annotation['event'] = int(event.lstrip('Ev')) -1 
                    annotation['vid_seg_id'] = vid_name + '_' + str(annotation['event'])
                    
                    if annotation:
                        self.annotations.append(annotation)

       
    def getanno(self, annotation):
        anno = {}
        anno['vid_seg_id'] = annotation['vid_seg_id']
        try:
            vid_feat, xtf_vid_feat = get_vid_embeddings(annotation['vid_path'], annotation['event'])
        except Exception as e:
            #print(e)
            return None
        # if vid_feat is None:
        #     return None
        anno['vid_feat'] = vid_feat  # add another dict xtf_feat
        anno['xtf_vid_feat'] = xtf_vid_feat
        anno['verb_feat'] = get_verb_embeddings(annotation['verb'])
        anno['roles_feat'] = get_role_embeddings(annotation['role_dict'])
        anno['labels_feat'] = get_label_embeddings(annotation['role_dict'])
        anno['labels_id'] = get_label_ids(annotation['role_dict'])
        
        return anno

dataset = VidSituDataset(split='train')
#pkl_dir = '/data/dataset/VidSitu/data/clip_feat_vit-b32'
pkl_dir = './vidsitu_data/' + clip_model_version
if not os.path.exists(pkl_dir):
    os.mkdir(pkl_dir)
annos = []
for annotation in tqdm(dataset.annotations):
    pkl_name = os.path.join(pkl_dir, annotation['vid_seg_id'])
    if not os.path.exists(pkl_name):
        with open(pkl_name, 'wb') as f:
            anno = dataset.getanno(annotation)
            if anno:
                pickle.dump(anno, f, pickle.HIGHEST_PROTOCOL)
    