import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import six
import pandas as pd
from utils.logger import LOGGER
import random

import json
import torch
from horovod import torch as hvd
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer
import numpy as np

def itm_val_collate(inputs):
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]


def transform(e):
    # x = e[0]
    # y = e[1]

    res = {}
    print(e[0])
    res['input_ids'] = torch.cat([s[0]['input_ids'] for s in e], 0)
    res['position_ids'] = torch.cat([s[0]['position_ids'] for s in e], 0)
    res['img_feat'] = torch.cat([s[0]['img_feat'] for s in e], 0)
    res['img_pos_feat'] = torch.cat([s[0]['img_pos_feat'] for s in e], 0)
    res['attn_masks'] = torch.cat([s[0]['attn_masks'] for s in e], 0)
    res['gather_index'] = torch.cat([s[0]['gather_index'] for s in e], 0)

    y = torch.cat([s[1] for s in e], 0)

    return res, y


# batch['input_ids'] = torch.tensor([input_ids])
# batch['position_ids'] = torch.tensor([position_ids])
# batch['img_feat'] = torch.tensor([img_feat], dtype=torch.float)
# batch['img_pos_feat'] = torch.tensor([img_pos_feat], dtype=torch.float)
# batch['attn_masks'] = torch.tensor([attn_masks])
# batch['gather_index'] = torch.tensor([gather_index])    
        


class MemeAIDataset(Dataset):
    def __init__(self, 
                    json_path = '',
                    npz_folder = '', 
                    transform=None, 
                    mode = 'train' #train valid test 
                    ):
        
        self.load_data(train_path = json_path, npz_folder=npz_folder)
        
        self.transform = transform
        
        indices = np.arange(len(self.images))
        if mode == 'train':
            random.shuffle(indices)

        self.indices = indices
        self.train = self.labels is not None
    
    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return  [self.get_example_wrapper(i) for i in six.moves.range(current, stop, step)]

        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]

        else:
            return self.get_example_wrapper(index)
    
    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)
    
    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)

        # if self.transform:
        #     example = self.transform(example)
        return example

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformations
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x

    def load_data(self, train_path = '/root/meme/train.json', npz_folder = '/root/output_meme_butd'):

        LOGGER.info('Load data from {}'.format(train_path))


        def bert_tokenize(tokenizer, text):
            ids = []
            for word in text.strip().split():
                ws = tokenizer.tokenize(word)
                if not ws:
                    # some special char
                    continue
                ids.extend(tokenizer.convert_tokens_to_ids(ws))
            return ids

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        json_files =open(train_path, "r")
        json_files = json_files.read().split('\n')
        json_files = [json.loads(i) for i in json_files]

        images = []
        labels = []
        for json_file in tqdm(json_files):

            batch = {}
            input_ids = bert_tokenize(tokenizer, json_file['text'])
            # print(input_ids)
            position_ids = range(len(input_ids))
            
            if json_file['id'] < 10000:
                id = '0' + str(json_file['id'])
            else:
                id = json_file['id']

            img_npz = np.load('{}/nlvr2_{}.npz'.format(npz_folder, id))
            img_feat = img_npz['features']
            img_pos_feat = np.concatenate((img_npz['norm_bb'], img_npz['conf']), axis=1)
            attn_masks = [1] * (len(input_ids)  + len(img_feat))
            gather_index = range(len(input_ids)  + len(img_feat))

            batch['input_ids'] = torch.tensor([input_ids])
            batch['position_ids'] = torch.tensor([position_ids])
            batch['img_feat'] = torch.tensor([img_feat], dtype=torch.float)
            batch['img_pos_feat'] = torch.tensor([img_pos_feat], dtype=torch.float)
            batch['attn_masks'] = torch.tensor([attn_masks])
            batch['gather_index'] = torch.tensor([gather_index])    
            
            images.append(batch)
            labels.append(torch.tensor(json_file['label'], dtype=torch.float))
        
        self.images = np.array(images)
        self.labels = np.array(labels)


        LOGGER.info('Loaded {} data from '.format(len(self.labels)))