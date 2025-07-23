import os
from random import randint

import argparse 
from tqdm import tqdm
import torch
import yaml

from model import build_rwkv_model, build_loop_model, build_transformer_model
from data_utils import get_data_sampler, Curriculum, get_task_sampler

import wandb

from eval import get_run_metrics

from torch.optim.lr_scheduler import StepLR, ExponentialLR
import math

torch.backends.cudnn.benchmark = True # Enable benchmark mode for faster training on GPUs with fixed input sizes

def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    # print("xs shape:", xs.shape, "ys shape:", ys.shape)
    # print("xs dtype:", xs.dtype, "ys dtype:", ys.dtype)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output, _ = model(xs, ys)

    output = output.to(torch.float32)
    ys = ys.to(torch.float32)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def train(model, config):
    lr = float(config['training']['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    scheduler_config = config['training'].get('lr_scheduler', {})
    min_lr = scheduler_config.get('min_learning_rate')
    if min_lr is not None:
        min_lr = float(min_lr)

    if scheduler_config:
        scheduler_type = scheduler_config['type']
        if scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'exponential':
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'cosine':
            # 实现余弦衰减逻辑
            max_steps = config['training']['train_steps']
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: 0.5 * (1 + math.cos(math.pi * step / max_steps))
            )
    curriculum = Curriculum(config['training']['curriculum'])
    starting_step = 0
    state_path = os.path.join(config['out_dir'], "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = config['model']['vocab_size'] # data dimensions, indicating the difficulty of the task
    bsize = config['training']['batch_size']
    data_sampler = get_data_sampler(config['training']['data'], n_dims=n_dims)
    print("Data sampler information:")
    print("  - data n_dims:", data_sampler.n_dims)
    print("  - data scale:", data_sampler.scale)
    print("  - data bias:", data_sampler.bias)
    # data = data_sampler.sample_xs(b_size=bsize, n_points=1000)
    # print("Data sample shape:", data.shape)
    task_sampler = get_task_sampler(
        config['training']['task'],
        n_dims,
        bsize,
        num_tasks=config['training']['num_tasks'],
        # config['training']['task_kwargs'],
    )

    print("Task sampler information:")
    print("  - task scale:", task_sampler.scale if hasattr(task_sampler, 'scale') else None)
    print("  - task w_b:", task_sampler.w_b if hasattr(task_sampler, 'w_b') else None)

    process_bar = tqdm(range(starting_step, config['training']['train_steps']))
    num_training_examples = config['training']['num_training_examples']
    current_lr = lr

    for i in process_bar:
        data_sampler_args = {}
        task_sampler_args = {}
        if "sparse" in config['training']['task']:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            n_points=curriculum.n_points,
            b_size=bsize,
            n_dims_truncated=curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        
        loss_func = task.get_training_metric()

        running_device = model.device
        xs = xs.to(running_device)
        ys = ys.to(running_device)
        loss, output = train_step(
            model, 
            xs, 
            ys, 
            optimizer, 
            loss_func
        )

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            # current_lr = optimizer.param_groups[0]['lr'] if scheduler else current_lr
            current_lr = optimizer.param_groups[0]['lr']
            if min_lr is not None and current_lr < min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr
                current_lr = min_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        if i % config['wandb']['log_every_steps'] == 0 and not config['test_run']:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "learning_rate": current_lr,
                },
                step=i,
            )

        curriculum.update()
        process_bar.set_description(f"loss {loss:.8f} | current_lr {current_lr:.4e}")

        if i % config['training']['save_every_steps'] == 0 and not config['test_run']:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None
            }
            torch.save(training_state, state_path)

        if (
            config['training']['keep_every_steps'] > 0
            and i % config['training']['keep_every_steps'] == 0
            and not config['test_run']
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RWKV model")
    parser.add_argument("--config", type=str, default="configs/toy.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Configurations:", config)
    # Initialize model
    model_config = config['model']
    print("Model configuration:", model_config)

    if model_config.get('loop_strategy') is not None:
        print("Using loop model configuration.")
        model = build_loop_model(model_config)
    else:
        print("Using standard model configuration.")    
        model = build_model(model_config)

    # model = build_model(config['model'])
    # model = build_loop_model(model_config)
    
    print("Model initialized successfully.")
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")

    

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    
    with open(os.path.join(config['out_dir'], "config.yaml"), "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    # Initialize Weights & Biases
    wandb.init(
            dir=config['out_dir'],
            project=config['wandb']['project'],
            config=config,
            notes=config['wandb']['notes'],
            name=config['wandb']['name'],
            resume=True,
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16)
    model.to(device)
    model.train()

    # Start training
    train(model, config)

    # Precompute eval metrics
    _ = get_run_metrics(config['out_dir'])