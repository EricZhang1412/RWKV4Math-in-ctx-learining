import gc
import importlib
import math
import os
import pdb, types, time, re, random
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.strategies import DeepSpeedStrategy
from rwkvfla.modules.token_shift import token_shift
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from rwkvfla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from base_models import NeuralNetwork, ParallelNetworks

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def __nop(ob):
    return ob
    
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyFunction = torch.compile

def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

HEAD_SIZE = 64
CHUNK_LEN = 256
############cuda kernels wkv7 for training############
flags = [
    "-res-usage",
    f"-D_C_={HEAD_SIZE}",
    f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math",
    "-O3",
    "-Xptxas -O3",
    "--extra-device-vectorization",
]
load(
    name="wind_backstepping",
    sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
    is_python_module=False,
    verbose=True,
    extra_cuda_cflags=flags,
)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        # assert T % CHUNK_LEN == 0
        #### check dtype and trun the variables into bf16
        if w.dtype != torch.bfloat16:
            w = w.bfloat16()
        if q.dtype != torch.bfloat16:
            q = q.bfloat16()
        if k.dtype != torch.bfloat16:
            k = k.bfloat16()
        if v.dtype != torch.bfloat16:
            v = v.bfloat16()
        if z.dtype != torch.bfloat16:
            z = z.bfloat16()
        if b.dtype != torch.bfloat16:
            b = b.bfloat16()
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(
            B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
        )
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(
            w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
        )
        return dw, dq, dk, dv, dz, db

def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)

############### Time-Mixing Module with params sharing ###############
class RWKV_Tmix_x070_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        num_all_layers = args.num_hidden_groups * args.inner_group_num
        with torch.no_grad():
            ratio_0_to_1 = group_id * loops_per_group / (num_all_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (group_id * loops_per_group / num_all_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) #!!! 0.5 comes from F.softplus!!!
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))
            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) #!!! notice eps value!!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
            del www, zigzag, linear, ddd
    
    @MyFunction
    def _forward_impl(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        # equals to:
        # xx = token_shift(x)

        xr = torch.addcmul(x, xx, self.x_r)
        xw = torch.addcmul(x, xx, self.x_w)
        xk = torch.addcmul(x, xx, self.x_k)
        xv = torch.addcmul(x, xx, self.x_v)
        xa = torch.addcmul(x, xx, self.x_a)
        xg = torch.addcmul(x, xx, self.x_g)
        # equivalent to:
        # xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.group_id * self.loops_per_group == 0:
            v_first = v  # store the v of the first layer
        else:
            v = torch.lerp(
                v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            )  # add value residual
        
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a-1) * self.k_a) 
        # equivalent to:
        # k = fused_k_rwkv7(k, a, self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = x.float()
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first
    def forward(self, x, v_first):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, v_first, use_reentrant=False)
        else:
            return self._forward_impl(x, v_first)

############RWKV_CMix_x070_v2: shared layers############
class RWKV_CMix_x070_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        num_all_layers = args.num_hidden_groups * args.inner_group_num
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (group_id * loops_per_group / num_all_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @MyFunction
    def _forward_impl(self, x):
        # xx = token_shift(x)
        xx = self.time_shift(x) - x
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
    def forward(self, x):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

class Block_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.group_id * loops_per_group == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070_v2(args, group_id, loops_per_group)
        self.ffn = RWKV_CMix_x070_v2(args, group_id, loops_per_group)
    def forward(self, x, v_first):
        if self.group_id * self.loops_per_group == 0:
            x = self.ln0(x)
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class BlockGroup(nn.Module):
    def __init__(self, args, group_id):
        super().__init__()
        self.args = args
        self.rwkv_layers = nn.ModuleList(
                [
                    Block_v2(args, group_id, i) 
                        for i in range(args.inner_group_num)
                ]
            ) # inner_group_num layers per group

    def forward(
            self, 
            x, 
            v_first,
            output_x,
            output_v_first,
        ):
        layer_x_states = []
        layer_v_first_states = []
        for rwkv_layer in self.rwkv_layers:
            x_states, v_first_states = rwkv_layer(x, v_first)

            if output_x:
                layer_x_states.append(x_states)
            if output_v_first:
                layer_v_first_states.append(v_first_states)

        if output_x:
            return x_states, v_first_states, layer_x_states
        elif output_v_first:
            return x_states, v_first_states, layer_v_first_states
        else:
            return x_states, v_first_states

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

@torch.jit.ignore
def sample_repeat_layers(
        num_layers: int,
        min_repeat: int = 4,
        max_repeat: int = 4,
        repeat_prob: float = 1.0,
    ):
        """
        随机采样每一层是否要 repeat，以及重复多少次
        :param num_layers: 总共的 group 层数
        :param min_repeat: 最少重复次数（默认1）
        :param max_repeat: 最大重复次数
        :param repeat_prob: 每层有概率开启 repeat（否则使用默认1次）
        :return: dict[layer_idx] = repeat_times
        """
        repeat_layers = {}
        for i in range(num_layers):
            if random.random() < repeat_prob:
                repeat_times = random.randint(min_repeat, max_repeat)
                repeat_layers[i] = repeat_times
            # else 默认不加，表示只执行1次
        return repeat_layers

class RWKV_shared(pl.LightningModule):
# class RWKV_shared(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.name = f"RWKV-shared_embd={args.n_embd}_group={args.num_hidden_groups}_layer_per_group={args.inner_group_num}_head={args.head_size}"
        #####################################################
        self.rand_step = getattr(args, 'rand_step', 0)
        self.mean_recurrence = getattr(args, 'mean_recurrence', 1)
        self.mean_backprop_depth = getattr(args,'mean_backprop_depth', 1)
        self.sampling_scheme = getattr(args, 'sampling_scheme', 'none')
        self.lockstep_n = getattr(args,'lockstep_n', False)
        self.lockstep_k = getattr(args,'lockstep_k', False)

        self.injection_type = getattr(args, 'injection_type', 'none')

        # self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.emb = nn.Linear(args.vocab_size, args.n_embd) # for function regression we do not need Voc2Embed (take Long Int type inputs)
        self.rwkv_layer_groups = nn.ModuleList([BlockGroup(args, i) for i in range(args.num_hidden_groups)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        # self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.head = nn.Linear(args.n_embd, 1, bias=False) # for function regression only output one result
        if self.injection_type == "linear":
            self.input_injection_adapter_x = nn.Linear(
                args.n_embd * 2,
                args.n_embd,
                bias=True,
            )
            self.input_injection_adapter_v = nn.Linear(
                args.n_embd * 2,
                args.n_embd,
                bias=True,
            )

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(
        self, 
        idx,
        ys,
        output_x=False,
        output_v_first=False,
        inds=None,
    ):
        args = self.args
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
                
        zs = self._combine(idx, ys)

        # zs shape should be [Batch_size, 2 * points, dim]
        B, T, _ = zs.size()

        all_x_states = []
        all_v_first_states = []
        all_logits = []

        x = self.emb(zs)
        v_first = torch.empty_like(x)
        ########## Get the number of the rwkv_shared groups ##########
        num_hidden_groups = len(self.rwkv_layer_groups)
        num_inner_layers = len(self.rwkv_layer_groups[0].rwkv_layers)

        settings_num_hidden_groups = args.num_hidden_groups
        settings_num_inner_layers = args.inner_group_num
        assert num_hidden_groups == settings_num_hidden_groups, "The number of hidden groups does not match the settings."
        assert num_inner_layers == settings_num_inner_layers, "The number of inner layers does not match the settings."

        repeat_layers = {} # no looping
        # repeat_layers = sample_repeat_layers(num_layers=len(self.rwkv_layer_groups))
        total_steps = len(self.rwkv_layer_groups)  # 假设你用了所有 group

        for i in range(total_steps):
            repeat_count = repeat_layers.get(i, 1)  # 默认为1次

            for _ in range(repeat_count):
                x_states, v_first_states = self.rwkv_layer_groups[i](
                    x, v_first,
                    output_x=output_x,
                    output_v_first=output_v_first
                )

                if output_x:
                    all_x_states.append(x_states)
                if output_v_first:
                    all_v_first_states.append(v_first_states)

                if self.injection_type == "add":
                    x = x + x_states
                    v_first = v_first + v_first_states
                elif self.injection_type in ["linear", "ffn"]:
                    # x = self.input_injection_adapter_x(torch.cat([x, x_states], dim=-1))
                    x = self.input_injection_adapter_x(torch.cat([
                            F.layer_norm(x, x.shape[-1:]),
                            F.layer_norm(x_states, x_states.shape[-1:])
                        ], dim=-1))
                    # v_first = self.input_injection_adapter_v(torch.cat([v_first, v_first_states], dim=-1))
                    v_first = self.input_injection_adapter_v(torch.cat([
                            F.layer_norm(v_first, v_first.shape[-1:]),
                            F.layer_norm(v_first_states, v_first_states.shape[-1:])
                        ], dim=-1))

            # logits = self.head(self.ln_out(x))

            # logits = logits[:, ::2, 0]
            # all_logits.append(logits)

        x = self.ln_out(x)
        x = self.head(x)
        x = x[:, ::2, 0]
        # all_logits.append(x)

        total_repeat_times = sum(repeat_layers.values())
        
        return x

    def generate_init_weight(self):
        print(
            """
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(
                f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end=""
            )

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / (self.args.num_hidden_groups * self.args.inner_group_num)
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            elif os.environ["RWKV_FLOAT_MODE"] == "fp32":
                m[n] = m[n].float()
            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)

class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
