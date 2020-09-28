"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import json
import os
from os.path import exists
import pickle
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, ItmEvalDataset, itm_eval_collate)
from model.itm import UniterForImageTextRetrieval

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM
from utils.itm_eval import inference, itm_eval



import torch
from horovod import torch as hvd
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer
import numpy as np

import argparse
import os
from os.path import exists, join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm

from data import (PrefetchLoader, TxtTokLmdb, ImageLmdbGroup,
                  ItmRankDataset, itm_rank_collate,
                  ItmValDataset, itm_val_collate,
                  ItmEvalDataset, itm_eval_collate)
from model.itm import UniterForImageTextRetrieval
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM
from utils.itm_eval import evaluate


from data.MemeDataset import MemeAIDataset


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    print('fasfafs: ', n_gpu)
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.train_config is not None:
        train_opts = Struct(json.load(open(opts.train_config)))
        opts.conf_th = train_opts.conf_th
        opts.max_bb = train_opts.max_bb
        opts.min_bb = train_opts.min_bb
        opts.num_bb = train_opts.num_bb
        
    
    # Prepare model
    checkpoint = torch.load(opts.checkpoint)
    model = UniterForImageTextRetrieval.from_pretrained(
        opts.model_config, checkpoint, img_dim=IMG_DIM)
    if 'rank_output' not in checkpoint:
        model.init_output()  # zero shot setting


    save_training_meta(opts)
    pbar = tqdm(total=opts.num_train_steps)

    model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
    add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    # store ITM predictions
    os.makedirs(join(opts.output_dir, 'results_val'))
    os.makedirs(join(opts.output_dir, 'results_test'))
    os.makedirs(join(opts.output_dir, 'results_train'))




    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')


    # load DBs and image dirs

    #create train_loader

    train_dataset = MemeAIDataset(json_path = '',
                                    npz_folder = '', 
                                    mode = 'train')

    #create dev loader
    val_dataset = MemeAIDataset(json_path = '',
                                    npz_folder = '', 
                                    mode = 'val')


    # npz_path = os.listdir('/root/output_meme_butd')
    # npz_path = ['/root/output_meme_butd/' + i for i in npz_path]
    json_files =open("/root/meme/train.json", "r")
    json_files = json_files.read().split('\n')
    json_files = [json.loads(i) for i in json_files]

    LOGGER.info('load {} file '.format(len(json_files)))

    eval_log, results = evaluate(model, json_files)

@torch.no_grad()
def evaluate(model, json_files):

    outfile = open('meme/train_feature.json','wb')

    res = []

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

    model.eval()
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    # score_matrix = inference(model, eval_loader)
    model.eval()

    if hvd.rank() == 0:
        pbar = tqdm(total=len(json_files))
    else:
        pbar = NoOp()

    for i, json_file in enumerate(json_files):
        
        if i == 10:break

        res_json ={'id': json_file['id']}

        batch = {}
        input_ids = bert_tokenize(tokenizer, json_file['text'])
        # print(input_ids)
        position_ids = range(len(input_ids))
        
        if json_file['id'] < 10000:
            id = '0' + str(json_file['id'])
        else:
            id = json_file['id']
        img_npz = np.load('/root/output_meme_butd/nlvr2_{}.npz'.format(id))
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
        
        
        # print('input_ids ', batch['input_ids'])
        # print('position_ids ', batch['position_ids'])
        # print('img_feat ', batch['img_feat'])
        # print('img_pos_feat ', batch['img_pos_feat'])
        # print('attn_masks ', batch['attn_masks'])
        # print('gather_index ', batch['gather_index'])
        # print()
        # print()
        # print()

        _, feature = model(batch, compute_loss=False)
        res_json['feature'] = feature.cpu().numpy()[0]
        res.append(res_json)

        pbar.update(1)

    model.train()
    pbar.close()


    pickle.dump(res,outfile)
    outfile.close()
    

    

    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds, ")
    # return eval_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="model checkpoint binary")
    parser.add_argument("--model_config", default=None, type=str,
                        help="model config json")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the inference results will be "
             "written.")

    # optional parameters
    parser.add_argument("--train_config", default=None, type=str,
                        help="hps.json from training (for prepro hps)")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')
    parser.add_argument("--batch_size", default=400, type=int,
                        help="number of tokens in a batch")

    # device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
