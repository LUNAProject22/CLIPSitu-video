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

from transformers import CLIPModel, CLIPTokenizer, AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

anno_dir = "/data/new_ssd/visual-comet/data/visualcomet"
image_dir = "/data/dataset/vcr/vcr1images"


class VizComDataset(object):
    
    def __init__(self, json_file):
        self.annotations = []
                
        self.annotations = self._read_data(json_file)

        self.embeddings = {}
        for i in tqdm(range(len(self.annotations))):
            encoding = self.getitem(i)
            if encoding['image_emb'] is not None:
                self.embeddings[encoding['image_id']] = encoding
            # if (i+1)%100 == 0:
            #     break

    def __len__(self):
        return len(self.annotations)
    
    def _read_data(self, json_file):
        annotations = []
        json_path = os.path.join(anno_dir, json_file)
        with open(json_path, "r") as f:
            img_list = json.load(f)
            count = 0
            for img_data in img_list:
                '''
                {
                    "img_fn": "lsmdc_3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER/3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER_00.27.43.141-00.27.45.534@0.jpg",
                    "movie": "3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER",
                    "metadata_fn": "lsmdc_3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER/3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER_00.27.43.141-00.27.45.534@0.json",
                    "place": "at a fancy party",
                    "event": "1 is trying to talk to the pretty woman in front of him",
                    "intent": ["ask the woman on a date", "get over his shyness"],
                    "before": ["approach 3 at an event", "introduce himself to 3",
                            "be invited to a dinner party", "dress in formal attire"],
                    "after": ["ask 3 to dance", "try to make a date with 3",
                            "greet her by kissing her hand", "order a drink from the server"]
                }
                '''
                
                 
                annotation = {}
                count += 1
                
                annotation['image'] = os.path.join(image_dir, img_data['img_fn'])
                annotation['name'] = img_data['img_fn']
                for k,v in img_data.items():
                    if k in ['place', 'event', 'intent', 'before', 'after']:
                        annotation[k] = v
                annotations.append(annotation)
                
                count += 1
        return annotations
    
    def getitem(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        
        image = Image.open(annotation['image'])
        image = image.convert("RGB")
        
        encoding ={}
        encoding['image_id'] = annotation['name']

        try:
            image_input = image_processor(images=image, return_tensors="pt").to(device)
            image_outputs = clip_model.vision_model.forward(**image_input)
            encoding['image_emb'] = clip_model.visual_projection(image_outputs['last_hidden_state']).detach().cpu()
            for k,v in annotation.items():
                if k in ['place', 'event']:
                    if not isinstance(v,list):
                        v = [v]
                    text_inputs = clip_tokenizer(text=v, return_tensors="pt", padding=True).to(device)
                    text_outputs = clip_model.text_model.forward(**text_inputs)
                    encoding[k] = text_outputs.pooler_output.detach().cpu()
                if k in ['intent', 'before', 'after']:
                    if not isinstance(v,list):
                        v = [v]
                    text_inputs = clip_tokenizer(text=v, return_tensors="pt", padding=True).to(device)
                    text_outputs = clip_model.text_model.forward(**text_inputs)
                    encoding[k+'_embed'] = text_outputs['last_hidden_state'].detach().cpu()
                    encoding[k] = v
        except:
            # print('blank_image ', annotation['image'])
            
            encoding['image_emb'] = None
            return encoding

        return encoding

train_dataset = VizComDataset('train_annots.json')
train_feat = train_dataset.embeddings
dev_dataset = VizComDataset('val_annots.json')
dev_feat = dev_dataset.embeddings
test_dataset = VizComDataset('test_annots.json')
test_feat = test_dataset.embeddings
    
save_folder = '../visualcomet_clip_xtf_features' #'/data/usrdata/roy/imsitu_clip_xtf_features'
with open(os.path.join(save_folder, 'xtf_visualcomet_b32.pkl'), 'wb') as f:
    # pickled = pickle.dumps(overall_feat, protocol=pickle.HIGHEST_PROTOCOL)
    # optimized_pickle = pickletools.optimize(pickled)
    # f.write(optimized_pickle)
    pickle.dump(train_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dev_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_feat, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
