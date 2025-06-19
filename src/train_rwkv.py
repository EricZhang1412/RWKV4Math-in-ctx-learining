from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

import datetime
import math
import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

from models_rwkv_x070 import RWKV_shared
from trainer import generate_init_weight, train_callback
from samplers import get_data_sampler
from tasks import get_task_sampler
from curriculum_rwkv import Curriculum

from eval_rwkv import get_run_metrics

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    curriculum = Curriculum(args)

    starting_step = 0
    state_path = os.path.join(args.proj_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = args.vocab_size
    bsize = args.batch_size
    data_sampler = get_data_sampler(args.training_data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        task_name=args.task_name,
        n_dims=n_dims,
        batch_size=bsize,
        pool_dict=None,
        num_tasks=args.training_num_tasks
    )
    pbar = tqdm(range(starting_step, args.train_steps))
    num_training_examples = args.num_training_examples
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.task_name:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        
        loss_func = task.get_training_metric()
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        curriculum.update()
        pbar.set_description(f"loss {loss}")
        if i % args.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

def main():
    parser = ArgumentParser()
    # parser.add_argument("--strategy", default="", type=str) ### already in pl.Trainer parser
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)
    ######## args for model architecture
    parser.add_argument("--n_embd", default=64, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--head_size", default=2, type=int)
    parser.add_argument("--vocab_size", default=5, type=int) #### raw data dimensions ()
    parser.add_argument("--num_hidden_groups", default=3, type=int)
    parser.add_argument("--inner_group_num", default=1, type=int)
    parser.add_argument("--injection_type", default=None, type=str)
    # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--grad_cp", default=0, type=int)

    parser.add_argument("--train_stage", default=1, type=int)
    # parser.add_argument("--precision", default="bf16", type=str) ### already in pl.Trainer parser
    # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--batch_size", default=0, type=int)
    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1,
                        type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)

    ##################### training args #####################
    parser.add_argument("--mean_recurrence", default=10, type=int) # mean recurrence steps per sample
    parser.add_argument("--mean_backprop_depth", default=5, type=int) # how many blocks to backprop
    parser.add_argument("--sampling_scheme", default='bptt', type=str) # how to sample recurrence steps
    parser.add_argument("--lockstep_n", default=False, type=bool) # 
    parser.add_argument("--lockstep_k", default=False, type=bool) # 
    parser.add_argument("--rand_step", default=0.0, type=int) # for convenience to change seed
    
    parser.add_argument("--mcleish_throttle", default=False, type=bool) # loss norm with n_grads steps
    parser.add_argument("--elbayad_weighing", default=True, type=bool) # weighted loss
    parser.add_argument("--elbayad_exponent", default=0.5, type=float) # weighted loss, with what power should future steps be penalized
    #########################################################
    ##################### Datasets args #####################
    parser.add_argument("--training_data", default="gaussian", type=str)
    parser.add_argument("--task_name", default="linear_regression", type=str)
    parser.add_argument("--training_num_tasks", default=1, type=int)
    parser.add_argument("--train_steps", default=5001, type=int)
    parser.add_argument("--num_training_examples", default=None, type=int)
    parser.add_argument("--save_every_steps", default=1000, type=int)
    parser.add_argument("--test_run", default=False, type=bool)

    parser.add_argument("--dims_start", default=5, type=int)
    parser.add_argument("--points_start", default=256, type=int)
    parser.add_argument("--dims_end", default=5, type=int)
    parser.add_argument("--points_end", default=256, type=int)
    parser.add_argument("--dims_inc", default=1, type=int)
    parser.add_argument("--points_inc", default=2, type=int)
    parser.add_argument("--dims_interval", default=2000, type=int)
    parser.add_argument("--points_interval", default=2000, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if "deepspeed" in args.strategy:
        import deepspeed

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 *
                           32)  # default = 3.5x emb size

    ######### create the workspace directory if not exists #########
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(args.proj_dir):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
        list_p.sort()
        max_p = list_p[-1]
        if len(list_p) > 1:
            args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
    
        if max_p == -1:
            args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        else:
            args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            if args.warmup_steps < 0:
                args.warmup_steps = 10
        args.epoch_begin = max_p + 1

    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info(
            "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")
    
    # assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    if args.precision == "fp32":
        rank_zero_info(
            "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    model = RWKV_shared(args)
    model.float()
    ################# Stage One: Initialization ################# 
    ################# Remember set --load_model to Null (--load_model "") #################
    if len(args.load_model) == 0 or args.train_stage == 1: 
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name
    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.', '')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.train_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")
    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    model.load_state_dict(load_dict)
    ################### Check Layer Shapes ###################
    for n in model.state_dict():
        shape = model.state_dict()[n].shape
        s0 = str(shape[0]) if len(shape) > 0 else ""
        s1 = str(shape[1]) if len(shape) > 1 else ""
        s2 = str(shape[2]) if len(shape) > 2 else ""
        s3 = str(shape[3]) if len(shape) > 3 else ""
        print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")
    
    model.cuda()
    model.train()
    train(model, args)

    _ = get_run_metrics(args.proj_dir, args)  # precompute metrics for eval
    
if __name__ == "__main__":
    main()
