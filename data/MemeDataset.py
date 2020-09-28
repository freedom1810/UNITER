import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import six
import pandas as pd
from utils.logger import LOGGER
import random

class MemeAIDataset(Dataset):
    def __init__(self, 
                    json_path = '',
                    npz_folder = '', 
                    transform=None, 
                    mode = 'train' #train valid test 
                    ):
        
        self.path = path

        self.load_data()
        
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

            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]

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

        if self.transform:
            example = self.transform(example)
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

    def load_data(train_path = '/root/meme/train.json', npz_folder = '/root/output_meme_butd'):

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
        for i, json_file in enumerate(json_files):

            batch = {}
            input_ids = bert_tokenize(tokenizer, json_file['text'])
            # print(input_ids)
            position_ids = range(len(input_ids))
            
            if json_file['id'] < 10000:
                id = '0' + str(json_file['id'])
            else:
                id = json_file['id']

            img_npz = np.load('{}}/nlvr2_{}.npz'.format(npz_folder, id))
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
            labels.append(json_file['label'])
        
        self.images = np.array(images)
        self.labels = np.array(labels)


        LOGGER.info('Loaded {} data from '.format(len(self.labels)))