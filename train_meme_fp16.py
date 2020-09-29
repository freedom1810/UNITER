"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for Image-Text Retrieval
"""
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
from model.meme import Meme
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM
from utils.itm_eval import evaluate

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from data.MemeDataset import MemeAIDataset
import numpy as np

def collate_fn(inputs):
    res = {}
    res['input_ids'] = torch.cat([s[0]['input_ids'] for s in inputs], 0)
    res['position_ids'] = torch.cat([s[0]['position_ids'] for s in inputs], 0)
    res['img_feat'] = torch.cat([s[0]['img_feat'] for s in inputs], 0)
    res['img_pos_feat'] = torch.cat([s[0]['img_pos_feat'] for s in inputs], 0)
    res['attn_masks'] = torch.cat([s[0]['attn_masks'] for s in inputs], 0)
    res['gather_index'] = torch.cat([s[0]['gather_index'] for s in inputs], 0)

    y = torch.cat([torch.tensor([s[1]]) for s in inputs], 0)

    return res, y
    # assert len(inputs) == 1, "input batch size > 1"
    # return inputs

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = opts.train_batch_size if is_train else 1
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader



def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank

    
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
    device = torch.device("cuda:1")
    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if hvd.rank() == 0:
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        os.makedirs(join(opts.output_dir, 'ckpt'))
        save_training_meta(opts)
        # TB_LOGGER.create(join(opts.output_dir, 'log'))
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        # store ITM predictions
        os.makedirs(join(opts.output_dir, 'results_val'))
        os.makedirs(join(opts.output_dir, 'results_test'))
        os.makedirs(join(opts.output_dir, 'results_train'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    train_dataset = MemeAIDataset(json_path = '/home/data/meme_json/train.json',
                                    npz_folder = '/home/data/faster_cnn_feature/', 
                                    mode = 'train')
    train_loader =  DataLoader(train_dataset, 
                                    batch_size = opts.train_batch_size, 
                                    shuffle = True, 
                                    num_workers = opts.n_workers,
                                    collate_fn=collate_fn)
    train_loader = PrefetchLoader(train_loader)

    # val
    val_dataset = MemeAIDataset(json_path = '/home/data/meme_json/dev.json',
                                    npz_folder = '/home/data/faster_cnn_feature/', 
                                    mode = 'val')
    val_loader =  DataLoader(val_dataset, 
                                    batch_size = opts.inf_minibatch_size, 
                                    shuffle = False, 
                                    num_workers = opts.n_workers,
                                    collate_fn=collate_fn)
    val_loader = PrefetchLoader(val_loader)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    model = Meme.from_pretrained(
        opts.model_config, state_dict=checkpoint,
        img_dim=IMG_DIM)
    model.init_output()  # pretrain ITM head is different from ranking head
    model.to(device)

    # make sure every process has same model parameters in the beginning
    # broadcast_tensors([p.data for p in model.parameters()], 0)
    # set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    # LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    # LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()

    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    # while True:
    for epoch in range(opts.epoch):
        print('epoch {}/ {}'.format(epoch, opts.epoch))
        pbar = tqdm(total=len(train_loader))

        model.train()
        preds = None
        gt = None

        for step, batch in enumerate(train_loader):
            x = batch[0]
            y = batch[1]
            n_examples += x['input_ids'].size(0)

            pred = model(x)

            if preds is None:

                preds = torch.sigmoid(pred)
                gt = y
            else:
                preds = torch.cat((preds, torch.sigmoid(pred)), dim = 0)
                gt = torch.cat((gt, y), dim = 0)


            loss = F.binary_cross_entropy(torch.sigmoid(pred), y)

            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()


        global_step += 1

        # learning rate scheduling
        lr_this_step = get_lr_sched(global_step, opts)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
        TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

        # log loss
        # NOTE: not gathered across GPUs for efficiency
        TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
        TB_LOGGER.step()

        # update model params
        if opts.grad_norm != -1:
            grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                        opts.grad_norm)
            TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            preds = preds.cpu().numpy().reshape(len(preds), )
            gt = gt.cpu().numpy()
            roc = roc_auc_score(gt, preds)
            acc = accuracy_score(gt, np.around(preds)) 
        train_log = {'train/roc': roc, 'train/acc': acc}
        TB_LOGGER.log_scaler_dict({f"train/{k}": v for k, v in train_log.items()})

        # monitor training throughput

        val_log = validate(model, val_loader)
        TB_LOGGER.log_scaler_dict({f"valid/{k}": v for k, v in val_log.items()})

        LOGGER.info(train_log)
        LOGGER.info(val_log)

        model_saver.save(model, global_step)

        pbar.close()


@torch.no_grad()
def validate(model, val_loader):
    # if hvd.rank() == 0:
    pbar = tqdm(total=len(val_loader))
    # else:
    #     pbar = NoOp()

    LOGGER.info("start running Image Retrieval validation ...")
    model.eval()
    n_ex = 0
    st = time()
    preds = None
    gt = None

    for x, y in val_loader:

        pred = model(x)
        if preds is None:

            preds = torch.sigmoid(pred)
            gt = y
        else:
            preds = torch.cat((preds, torch.sigmoid(pred)), dim = 0)
            gt = torch.cat((gt, y), dim = 0)

        pbar.update(1)


    preds = preds.cpu().numpy().reshape(len(preds), )
    gt = gt.cpu().numpy()
    roc = roc_auc_score(gt, preds)
    acc = accuracy_score(gt, np.around(preds)) 
    val_log = {'valid/roc': roc,
               'valid/acc': acc}

    pbar.close()
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--negative_size", default=1, type=int,
                        help="Number of negative samples per positive sample")
    parser.add_argument("--inf_minibatch_size", default=400, type=int,
                        help="batch size for running inference. "
                             "(used for validation, and evaluation)")

    parser.add_argument("--epoch", default=50, type=int,
                        help="epoch")
    parser.add_argument("--epoch_freeze", default=10, type=int,
                        help="epoch")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=0.25, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--full_val', action='store_true',
                        help="Always run full evaluation during training")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')
    parser.add_argument('--model_config')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
