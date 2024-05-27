import collections
import os
import os.path as op
from model.build_finetune import build_finetune_model
import torch
import numpy as np
import random
import time
import torch.nn as nn

from datasets.build import build_mix_loader, build_zero_shot_loader
from model.clip_model_ms import build_CLIP_from_openai_pretrained


from utils.iotools import save_train_configs
from utils.logger import setup_logger

# from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from mindspore import context


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    args = get_args()
    set_seed(args.seed)
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    trainset ,train_loader, val_img_loader0, val_txt_loader0, val_img_loader1, val_txt_loader1, val_img_loader2, val_txt_loader2, num_classes = build_zero_shot_loader(args,finetune=True)
    model = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
    
    
    evaluator0 = Evaluator(val_img_loader0, val_txt_loader0)
    evaluator1 = Evaluator(val_img_loader1, val_txt_loader1)
    evaluator2 = Evaluator(val_img_loader2, val_txt_loader2)

    evaluator0.eval(model)
