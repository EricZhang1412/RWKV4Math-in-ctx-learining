from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

def valid_row(r):
    return r.task == task

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "./output_loop_1_G3L1"
df = read_run_dir(run_dir)
print(df)

task = "linear_regression"
# run_id = "c0cd5da0-9165-42f1-bcc2-d8fcc8049137"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task)

# recompute_metrics = False    

# if recompute_metrics:
#     get_run_metrics(run_path)  # these are normally precomputed at the end of training

print(f"run directory:{run_dir}")
metrics = collect_results(run_dir, df, valid_row=valid_row)

_, conf = get_model_from_run(run_path, only_conf=True)
n_dims = conf['model']['vocab_size']

models = relevant_model_names[task]

image_dir = os.path.join(run_dir, "figures")  # 默认当前目录
os.makedirs(image_dir, exist_ok=True)

fig, ax = basic_plot(metrics["standard"], models=models)
#### save the image
file_name = os.path.join(image_dir, f"{task}_performance_comparison.png")
fig.savefig(file_name, bbox_inches='tight')


# plot any OOD metrics
for name, metric in metrics.items():
    print(f"name: {name}")
    print(f"metric: {metric}")

    # if name == "standard": continue
   
    if "scale" in name:
        scale = float(name.split("=")[-1])**2
    else:
        scale = 1.0

    trivial = 1.0 if "noisy" not in name else (1+1/n_dims)
    fig, ax = basic_plot(metric, models=models, trivial=trivial * scale)
    ax.set_title(name)
    
    if "ortho" in name:
        ax.set_xlim(-1, n_dims - 1)
    ax.set_ylim(-.1 * scale, 1.5 * scale)

    file_name = os.path.join(image_dir, f"{name}_performance_comparison.png")
    fig.savefig(file_name, bbox_inches='tight')

