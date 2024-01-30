import torch
import json
import os
import pickle

from torch.nn.functional import softmax
import av
import numpy as np
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

import torch.nn.functional as F
torch.manual_seed(42)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import CLIPModel, CLIPTokenizer, AutoProcessor, CLIPTextModel

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

noun2id = pickle.load(open('noun2id.pkl','rb'))


def get_vid_embeddings(vid_path): # clip embeddings for every 2 sec video
    
    frames = os.listdir(vid_path)
    
    xtf_vid_features = []
    vid_features = []
    for frame in frames: # 11 frames
        image = Image.open(os.path.join(vid_path, frame))
        image_input = image_processor(images=image, return_tensors="pt").to(device)
        outputs =  clip_model.vision_model.forward(**image_input)
        xtf_image_emb = clip_model.visual_projection(outputs['last_hidden_state']).detach().cpu()   
        image_emb = clip_model.visual_projection(outputs['pooler_output']).detach().cpu()

        xtf_vid_features.append(xtf_image_emb)
        vid_features.append(image_emb)
    
    xtf_vid_features = torch.stack(xtf_vid_features).squeeze()
    vid_features = torch.stack(vid_features).squeeze()
    
    return xtf_vid_features, vid_features

def get_verb_embeddings(verb):
    verb_tok = clip_tokenizer(verb, padding=True, return_tensors="pt")
    verb_feat = clip_text_model(**verb_tok)['pooler_output'].detach().cpu()
    
    return verb_feat

def get_role_embeddings(role_dict):
    
    roles_feat = []
    for role in role_dict.keys():
        role_tok = clip_tokenizer(role, padding=True, return_tensors="pt")
        role_feat =  clip_text_model(**role_tok)['pooler_output'].detach().cpu()
        roles_feat.append(role_feat)
    
    return torch.stack(roles_feat).squeeze()


def get_label_ids(role_dict):
    labels_id = []
    for role in role_dict.keys():
        id = noun2id[role_dict[role]]
        labels_id.append(id)
    
    return labels_id

class VidSituDataset(object):
    def __init__(self): # split is train or valid
        root_dir = "/data/dataset/VidSitu/data/"
        
        split_dir = os.path.join(root_dir, "vidsitu_annotations/split_files")
        split = 'train'
        split_name = os.path.join(split_dir, 'vseg_split_' + split + '_lb.json')
        
        anno_dir = os.path.join(root_dir, "vidsitu_annotations/vseg_ann_files")
        anno_name = os.path.join(anno_dir, 'vsann_' + split + '_lb.json')
        annos = json.load(open(anno_name,'r'))
        
        self.vids = json.load(open(split_name,'r'))
        split = 'valid'                      
        split_name = os.path.join(split_dir, 'vseg_split_' + split + '_lb.json')
        self.vids.extend(json.load(open(split_name,'r')))

        anno_name = os.path.join(anno_dir, 'vsann_' + split + '_lb.json')
        annos.extend(json.load(open(anno_name,'r')))
        

        self.vid_trimmed_dir = os.path.join(root_dir, "vsitu_video_frames_dir", "vsitu_11_frames_per_vid")
        
        
        self.annotations = []
        for anno in annos:
            annotation = {}
            vid_name = anno['Ev1']['vid_seg_int']
            # sanity check for existence of video frames
            if os.path.exists(os.path.join(self.vid_trimmed_dir, vid_name)):
                annotation['vid_seg_id'] = vid_name
                annotation['verb'] = []
                annotation['role_dict'] = []
                for event in anno.keys(): # should return Ev1,...,Ev5
                    event_info = anno[event]
                    annotation['vid_path'] = os.path.join(self.vid_trimmed_dir, vid_name)
                    annotation['verb'].append(event_info['Verb'].split()[0])
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
                    annotation['role_dict'].append(role_dict)
                self.annotations.append(annotation)

       
    def getanno(self, annotation):
        anno = {}
        anno['vid_id'] = annotation['vid_seg_id']
        xtf_vid_feat, vid_feat = get_vid_embeddings(annotation['vid_path'])
        if vid_feat is None:
            return None
        anno['xtf_vid_feat'] = xtf_vid_feat
        anno['vid_feat'] = vid_feat
        verb_feat = []
        for verb in annotation['verb']:
            verb_feat.append(get_verb_embeddings(verb))
        anno['verb_feat'] = torch.stack(verb_feat)
        role_feat = []
        for roles in annotation['role_dict']:
            role_feat.append(get_role_embeddings(roles))
        anno['roles_feat'] = role_feat
        
        return anno

dataset = VidSituDataset()
pkl_dir = '/data/dataset/VidSitu/data/clip_feat_vit-b32_11f'
annos = []
for annotation in tqdm(dataset.annotations):
    pkl_name = os.path.join(pkl_dir, annotation['vid_seg_id'])
    if not os.path.exists(pkl_name):
        with open(pkl_name, 'wb') as f:
            anno = dataset.getanno(annotation)
            if anno:
                pickle.dump(anno, f, pickle.HIGHEST_PROTOCOL)
    