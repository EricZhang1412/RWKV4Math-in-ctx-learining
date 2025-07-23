# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from rwkvfla.layers.attn import Attention
from rwkvfla.layers.rwkv7 import RWKV7Attention
from rwkvfla.models.transformer.configuration_transformer import TransformerConfig
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from rwkvfla.models.utils import Cache
from rwkvfla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, LayerNorm
from rwkvfla.modules.activations import ACT2FN
from rwkvfla.modules import GatedMLP as TransformerMLP
from rwkvfla.modules import RMSNorm
from rwkvfla.modules.l2warp import l2_warp
from rwkvfla.modules.token_shift import token_shift

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

from base_models import NeuralNetwork, ParallelNetworks
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import tree
import xgboost as xgb

logger = logging.get_logger(__name__)

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
    # print(models)
    return models

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

# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
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


class RWKV7FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'sqrelu',
        layer_idx: int = None,
        num_hidden_layers: int = None,
    ) -> RWKV7FeedForward:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.zeros(hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for name, module in self.named_modules():
            module._in_rwkv_module = True

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, RWKV7FeedForward):
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (module.layer_idx / module.num_hidden_layers)  # 1 to ~0
                ddd = torch.ones(1, 1, module.hidden_size)
                for i in range(module.hidden_size):
                    ddd[0, 0, i] = i / module.hidden_size
                module.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0**4).squeeze()

            # Initialize key and value weights as in CMix_x070
            torch.nn.init.orthogonal_(module.key.weight)
            module.value.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[Cache] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        if x.shape[1] == 1 and state is not None and state[self.layer_idx]['ffn_state'] is not None:
            shifted = state[self.layer_idx]['ffn_state'].unsqueeze(1)
            delta = shifted - x
        elif state is not None and state[self.layer_idx]['ffn_state'] is not None:
            shifted = self.time_shift(x)
            shifted[:, 0] = state[self.layer_idx]['ffn_state'][-1]
            delta = shifted - x
        else:
            delta = token_shift(x, cu_seqlens)
        if state is not None:
            # no need to update the offset twice
            state.update(ffn_state=x[:, -1], layer_idx=self.layer_idx, offset=0)
        return self.value(self.act_fn(self.key(x.addcmul(delta, self.x_k)))), state


class RWKV7Block(nn.Module):

    def __init__(
        self,
        config: RWKV7Config,
        layer_idx: int
    ) -> RWKV7Block:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        if config.norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps
            )
        self.attn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers
            )
        self.ffn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, past_key_values, v_first = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            v_first=v_first,
            cu_seqlens=cu_seqlens,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
        hidden_states, past_key_values = self.ffn(
            hidden_states, attention_mask, past_key_values, cu_seqlens, **kwargs
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, v_first)

        return outputs

class RWKV7BlockGroup(nn.Module):
    def __init__(
        self,
        config: RWKV7LoopConfig,
        group_idx: int
    ) -> RWKV7BlockGroup:
        super().__init__()

        self.config = config
        self.group_idx = group_idx
        self.blocks = nn.ModuleList(
            [RWKV7Block(config, layer_idx) for layer_idx in range(
                group_idx * config.num_layers_per_group,
                (group_idx + 1) * config.num_layers_per_group
            )]
        )
        #### Add fusion for loop computation in one block-group
        if config.loop_strategy == 'custom':
            self.group_loop_times = config.loop_times["group_idx"]
        if config.loop_strategy == 'uniform':
            self.group_loop_times = config.loop_times

        if self.group_loop_times > 1:
            if config.loop_injection == 'linear_norm':
                self.loop_injection_x = nn.Sequential(
                    nn.Linear(
                        2 * config.hidden_size, config.hidden_size, bias=False
                    ),
                    nn.LayerNorm(
                        config.hidden_size,
                        bias=config.norm_bias,
                        eps=config.norm_eps
                    ),
                )
                self.loop_injection_v = nn.Sequential(
                    nn.Linear(
                        2 * config.hidden_size, config.hidden_size, bias=False
                    ),
                    nn.LayerNorm(
                        config.hidden_size,
                        bias=config.norm_bias,
                        eps=config.norm_eps
                    ),
                )
            # elif config.loop_injection == 'residual':
            #     self.loop_injection_x = nn.Sequential(
            #         nn.Linear(
            #             config.hidden_size, config.hidden_size, bias=False
            #         ),
            #         nn.LayerNorm(
            #             config.hidden_size,
            #             bias=config.norm_bias,
            #             eps=config.norm_eps
            #         )
            #     )
            #     self.loop_injection_v = nn.Sequential(  
            #         nn.Linear(
            #             config.hidden_size, config.hidden_size, bias=False
            #         ),
            #         nn.LayerNorm(
            #             config.hidden_size,
            #             bias=config.norm_bias,
            #             eps=config.norm_eps
            #         )
            #     )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        attentions = None
        for i in range(self.group_loop_times):
            for block in self.blocks:
                previous_hidden_states = hidden_states
                previous_v_first = v_first
                block_output = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    cu_seqlens=cu_seqlens,
                    **kwargs
                )
                hidden_states, attentions, past_key_values, v_first = block_output
            if self.config.loop_injection == 'linear_norm' and i < self.group_loop_times - 1:
                # Inject the loop computation into the hidden states
                #concat hidden states with the previous hidden states
                hidden_states_cat = torch.cat((hidden_states, previous_hidden_states), dim=-1)
                #concat v_first with the previous v_first
                v_first_cat = torch.cat((v_first, previous_v_first), dim=-1)
                #apply linear layer
                float_hidden_states = hidden_states_cat.float()  # 转换为fp32
                injected_hidden_states = self.loop_injection_x(float_hidden_states)
                hidden_states = injected_hidden_states.to(hidden_states.dtype)
                float_v_first = v_first_cat.float()  # 转换为fp32
                injected_v_first = self.loop_injection_v(float_v_first)
                v_first = injected_v_first.to(v_first.dtype)
            elif self.config.loop_injection == 'residual' and i < self.group_loop_times - 1:
                # Inject the loop computation into the hidden states
                # by adding the previous hidden states to the current hidden states
                # import pdb; pdb.set_trace()
                hidden_states = hidden_states + previous_hidden_states
                float_hidden = hidden_states.float()  # 转换为fp32
                normed = torch.nn.functional.layer_norm(
                    float_hidden,
                    (self.config.hidden_size,)
                )
                hidden_states = normed.to(hidden_states.dtype)
                v_first = v_first + previous_v_first
                float_v_first = v_first.float()  # 转换为fp32
                normed_v_first = torch.nn.functional.layer_norm(
                    float_v_first,
                    (self.config.hidden_size,)
                )
                v_first = normed_v_first.to(v_first.dtype)
        
        outputs = (hidden_states, attentions, past_key_values, v_first)
        return outputs
    

class RWKV7PreTrainedModel(PreTrainedModel):

    config_class = RWKV7Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RWKV7Block']
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @torch.no_grad()
    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, nn.Embedding):
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L396C12-L399C58
            scale = -1e-4
            nn.init.uniform_(module.weight, a=scale, b=-scale)
        elif isinstance(module, nn.Linear) and hasattr(self, 'lm_head') and module is self.lm_head:
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L403
            if self.config.vocab_size > self.config.hidden_size:
                scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
            else:
                scale = 0.5
            original_dtype = module.weight.dtype
            module.weight.data = nn.init.orthogonal_(module.weight.data.to(torch.float32), gain=scale).to(original_dtype)
        # Init Attention parameters
        elif isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(module, '_in_rwkv_module', False) is False:
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters') and getattr(module, '_in_rwkv_module', False) is False:
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class RWKV7Model(RWKV7PreTrainedModel):

    def __init__(self, config: RWKV7Config):
        super().__init__(config)

        self.name = f"RWKV7Model_{config.num_hidden_layers}layers_{config.hidden_size}hidden"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embeddings = nn.Linear(
            config.vocab_size, config.hidden_size, bias=False
        )  # Using Linear instead of Embedding for compatibility with RWKV7 ICL
        self.layers = nn.ModuleList([RWKV7Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        self._read_out = nn.Linear(config.hidden_size, 1)

        self.gradient_checkpointing = False

        self.post_init()

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

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override the load_state_dict method to handle migration from version 1 to version 2.
        Handles hierarchical keys like 'model.layers.0.attn.x_x'.
        """
        # Collect all layer indices from the state_dict keys
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("model.layers."):
                # Extract the layer index from the key
                try:
                    layer_idx = int(key.split(".")[2])  # Extract the number after 'model.layers.'
                    layer_indices.add(layer_idx)
                except ValueError:
                    # Skip keys that don't match the expected format
                    continue

        # Sort the layer indices to process them in order
        sorted_layer_indices = sorted(layer_indices)

        # Migration logic for each layer
        for layer_idx in sorted_layer_indices:
            layer_prefix = f"model.layers.{layer_idx}"
            attn_prefix = f"{layer_prefix}.attn"

            # Check if the layer contains the old 'x_x' parameter
            if f"{attn_prefix}.x_x" in state_dict:
                logger.info(f"Migrating weights for layer {layer_idx} from RWKV7Attention version 1 to version 2...")
                # Extract the x_x parameter
                x_x = state_dict[f"{attn_prefix}.x_x"]
                with torch.no_grad():
                    # Create new parameters for version 2
                    state_dict[f"{attn_prefix}.x_r"] = x_x[0].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_w"] = x_x[1].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_k"] = x_x[2].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_v"] = x_x[3].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_a"] = x_x[4].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_g"] = x_x[5].unsqueeze(0).unsqueeze(0)

        # Call the parent method to load the modified state_dict
        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # If the parent method does not support `assign`, fall back to strict loading
            logger.warning(
                "`assign` parameter is not supported by the parent `load_state_dict` method. "
                "Falling back to default behavior."
            )
            super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        xs, ys, inds=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # if input_ids is None and inputs_embeds is None:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds = self.embeddings(input_ids)

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(zs)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values, v_first = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    v_first,
                    cu_seqlens,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values, v_first = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    cu_seqlens=cu_seqlens,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # print("Hidden states shape:", hidden_states.shape)
        prediction = self._read_out(hidden_states)
        # print("Prediction shape:", prediction.shape)
        pred_on_xs = prediction[:, ::2, 0][:, inds]
        # print("Prediction on xs shape:", pred_on_xs.shape)

        # Hidden states shape: torch.Size([64, 22, 64])
        # Prediction shape: torch.Size([64, 22, 1])
        # Prediction on xs shape: torch.Size([64, 11])

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return pred_on_xs, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )

def build_rwkv_model(config_dict):
    """
    Build the RWKV model based on the provided configuration.
    
    Args:
        config (RWKV7Config): Configuration for the RWKV model.
        
    Returns:
        RWKV7Model: An instance of the RWKV model.
    """
    ####### parse the configuration dictionary #######
    config = RWKV7Config(
        hidden_size=config_dict.get("hidden_size", 64),
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        head_dim=config_dict.get("head_dim", 64),
        decay_low_rank_dim=config_dict.get("decay_low_rank_dim", 64),
        gate_low_rank_dim=config_dict.get("gate_low_rank_dim", 128),
        a_low_rank_dim=config_dict.get("a_low_rank_dim", 64),
        v_low_rank_dim=config_dict.get("v_low_rank_dim", 16),
        max_position_embeddings=config_dict.get("max_position_embeddings", None),
        vocab_size=config_dict.get("vocab_size", 5)
    )
    return RWKV7Model(config=config)
    
class RWKV7LoopConfig(RWKV7Config):
    """
    Configuration for RWKV7LoopModel.
    Inherits from RWKV7Config and adds specific parameters for loop computation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #### Add loop strategy ####
        self.num_block_groups = kwargs.get('num_block_groups', 1)  # Number of block groups for loop computation
        self.num_layers_per_group = kwargs.get('num_layers_per_group', self.num_hidden_layers // self.num_block_groups)
        if self.num_layers_per_group * self.num_block_groups != self.num_hidden_layers:
            raise ValueError("num_layers_per_group * num_block_groups must equal num_hidden_layers")    
        self.loop_strategy = kwargs.get('loop_strategy', 'uniform')  
        # Strategy for looping over block groups:
        # 'uniform': All block groups are looped through the same number of times. The config function receives a single integer for the number of loops.
        # 'custom': Custom strategy where each block group can have a different number of loops. The config function receives a dictionary with the number of loops for each block group.
        self.loop_times = kwargs.get('loop_times', 0)  # Number of times to loop through each block group
        # print(f"Loop strategy: {self.loop_strategy}, Loop times: {self.loop_times}")
        if self.loop_strategy == 'custom' and self.loop_times is None:
            raise ValueError("loop_times must be provided when loop_strategy is 'custom'")
        if self.loop_strategy not in ['uniform', 'custom']:
            raise ValueError("loop_strategy must be either 'uniform' or 'custom'")
        if self.loop_strategy == 'custom' and len(self.loop_times) != self.num_block_groups:
            raise ValueError("loop_times must have the same length as num_block_groups when loop_strategy is 'custom'")
        if self.loop_strategy == 'uniform' and not isinstance(self.loop_times, int):
            raise ValueError("loop_times must be an integer when loop_strategy is 'uniform'")
        
        self.loop_injection = kwargs.get('loop_injection', 'linear_norm')  # Method to inject loop computation

####[TODO] Unreasoned buggy code for RWKV7PreTrainedLoopModel
# class RWKV7PreTrainedLoopModel(RWKV7PreTrainedModel):
#     """
#     A base class for RWKV7 models that supports loop computation.
#     Inherits from RWKV7PreTrainedModel and adds specific parameters for loop computation.
#     """
#     config_class = RWKV7LoopConfig
#     base_model_prefix = 'model'
#     supports_gradient_checkpointing = True
#     _no_split_modules = ['RWKV7BlockGroup']
#     # _no_split_modules = ['RWKV7Block']
#     _supports_cache_class = True
#     _skip_keys_device_placement = ["past_key_values"]
#     def __init__(self, *inputs, **kwargs):
#         super().__init__(*inputs, **kwargs)
        
#     @torch.no_grad()
#     def _init_weights(
#         self,
#         module: nn.Module,
#         rescale_prenorm_residual: bool = True,
#         num_residuals_per_layer: int = 2,
#     ):
#         if isinstance(module, nn.Embedding):
#             # Initialize embedding weights
#             scale = -1e-4
#             nn.init.uniform_(module.weight, a=scale, b=-scale)
#         elif isinstance(module, nn.Linear) and hasattr(self, 'lm_head') and module is self.lm_head:
#             # Initialize the output layer weights
#             if self.config.vocab_size > self.config.hidden_size:
#                 scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
#             else:
#                 scale = 0.5
#             original_dtype = module.weight.dtype
#             module.weight.data = nn.init.orthogonal_(module.weight.data.to(torch.float32), gain=scale).to(original_dtype)
#         elif hasattr(module, 'reset_parameters') and getattr(module, '_in_rwkv_module', False) is False:
#             module.reset_parameters()
        
#         if rescale_prenorm_residual:
#             # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme
#             p = None
#             if hasattr(module, 'o_proj'):
#                 p = module.o_proj.weight
#             elif hasattr(module, 'down_proj'):
#                 p = module.down_proj.weight
#             if p is not None:
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

class RWKV7LoopModel(RWKV7PreTrainedModel):
    """
    A model that loops over the RWKV7 block groups for each input point.
    """

    def __init__(self, config: RWKV7LoopConfig):
        super().__init__(config)
        self.name = f"RWKV7LoopModel_{config.num_hidden_layers}layers_{config.hidden_size}hidden"
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Linear(
            config.vocab_size, config.hidden_size, bias=False
        )
        self.block_groups = nn.ModuleList(
            [RWKV7BlockGroup(config, group_idx) for group_idx in range(config.num_block_groups)]
        )
        self.norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        self._read_out = nn.Linear(config.hidden_size, 1)
    
        self.gradient_checkpointing = False
        self.post_init()
    
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

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    #[TODO] Implement the load_state_dict method for RWKV7LoopModel if needed: def load_state_dict(self, state_dict, strict=True, assign=False):

    def forward(
        self,
        xs, ys, inds=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`RWKV7LoopModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys)
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(zs)
        
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        v_first = torch.zeros_like(hidden_states)
        
        ##### Add Loop computing over block groups #####
        for group in self.block_groups:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values, v_first = self._gradient_checkpointing_func(
                    group.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    v_first,
                    cu_seqlens,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values, v_first = group(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    cu_seqlens=cu_seqlens,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        prediction = self._read_out(hidden_states)
        pred_on_xs = prediction[:, ::2, 0][:, inds]

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return pred_on_xs, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )

def build_loop_model(config_dict):
    """
    Build the RWKV loop model based on the provided configuration.
    
    Args:
        config (RWKV7LoopConfig): Configuration for the RWKV loop model.
        
    Returns:
        RWKV7LoopModel: An instance of the RWKV loop model.
    """
    ####### parse the configuration dictionary #######
    config = RWKV7LoopConfig(
        hidden_size=config_dict.get("hidden_size", 64),
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        head_dim=config_dict.get("head_dim", 64),
        decay_low_rank_dim=config_dict.get("decay_low_rank_dim", 64),
        gate_low_rank_dim=config_dict.get("gate_low_rank_dim", 128),
        a_low_rank_dim=config_dict.get("a_low_rank_dim", 64),
        v_low_rank_dim=config_dict.get("v_low_rank_dim", 16),
        max_position_embeddings=config_dict.get("max_position_embeddings", None),
        vocab_size=config_dict.get("vocab_size", 5),
        num_block_groups=config_dict.get("num_block_groups", 1),
        num_layers_per_group=config_dict.get("num_layers_per_group", None),
        loop_strategy=config_dict.get("loop_strategy", 'uniform'),
        loop_times=config_dict.get("loop_times", None),
        loop_injection=config_dict.get("loop_injection", 'linear_norm')
    )
    return RWKV7LoopModel(config=config)

################### Transformers Model Initialization ###################
# from rwkvfla.models import TransformerModel
from rwkvfla.models import TransformerConfig

class TransformerBlock(nn.Module):

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs

class TransformerPreTrainedModel(PreTrainedModel):

    config_class = TransformerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['TransformerBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

class TransformerModel(TransformerPreTrainedModel):

    def __init__(
        self,
        config: TransformerConfig
    ) -> TransformerModel:
        super().__init__(config)
        self.name = f"Transformer_{config.num_hidden_layers}layers_{config.hidden_size}hidden"
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embeddings = nn.Linear(
            config.vocab_size, config.hidden_size, bias=False
        )  # Using Linear instead of Embedding for compatibility with ICL
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self._read_out = nn.Linear(config.hidden_size, 1)
        self.gradient_checkpointing = False

        self.post_init()

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

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        xs, ys, inds=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is None and inputs_embeds is None:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if use_cache and not isinstance(past_key_values, Cache):
        #     past_key_values = Cache.from_legacy_cache(past_key_values)

        # if inputs_embeds is None:
        #     inputs_embeds = self.embeddings(input_ids)

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(zs)
        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    **kwargs
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        prediction = self._read_out(hidden_states)
        pred_on_xs = prediction[:, ::2, 0][:, inds]
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return pred_on_xs, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )

def build_transformer_model(config_dict):
    """
    Build the transformer model based on the provided configuration.
    
    Args:
        config (TransformerConfig): Configuration for the transformer model.
        
    Returns:
        TransformerModel: An instance of the transformer model.
    """
    config = TransformerConfig(
        # hidden_size: int = 2048,
        # num_hidden_layers: int = 24,
        # num_heads: int = 32,
        # num_kv_heads: int = None,
        # qkv_bias: bool = False,
        # qk_norm: bool = False,
        # window_size: Optional[int] = None,
        # rope_theta: Optional[float] = 10000.,
        # max_position_embeddings: int = 2048,
        # hidden_ratio: Optional[int] = 4,
        # intermediate_size: Optional[int] = None,
        # hidden_act: str = "swish",
        # initializer_range: float = 0.02,
        # elementwise_affine: Optional[bool] = True,
        # norm_eps: float = 1e-6,
        # use_cache: bool = True,
        # pad_token_id: int = None,
        # bos_token_id: int = 1,
        # eos_token_id: int = 2,
        # tie_word_embeddings: bool = False,
        # fuse_norm: bool = True,
        # fuse_swiglu: bool = True,
        # fuse_cross_entropy: bool = True,
        # use_l2warp: bool = False,
        # vocab_size: int = 32000,
        hidden_size=config_dict.get("hidden_size", 64),
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        num_heads=config_dict.get("num_heads", 8),
        num_kv_heads=config_dict.get("num_kv_heads", None),
        qkv_bias=config_dict.get("qkv_bias", False),
        qk_norm=config_dict.get("qk_norm", False),
        window_size=config_dict.get("window_size", None),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        max_position_embeddings=config_dict.get("max_position_embeddings", 2048),
        hidden_ratio=config_dict.get("hidden_ratio", 4),
        intermediate_size=config_dict.get("intermediate_size", None),
        hidden_act=config_dict.get("hidden_act", "swish"),
        initializer_range=config_dict.get("initializer_range", 0.02),
        elementwise_affine=config_dict.get("elementwise_affine", True),
        norm_eps=config_dict.get("norm_eps", 1e-6),
        use_cache=config_dict.get("use_cache", True),
        pad_token_id=config_dict.get("pad_token_id", None),
        bos_token_id=config_dict.get("bos_token_id", 1),
        eos_token_id=config_dict.get("eos_token_id", 2),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
        fuse_norm=config_dict.get("fuse_norm", True),
        fuse_swiglu=config_dict.get("fuse_swiglu", True),
        fuse_cross_entropy=config_dict.get("fuse_cross_entropy", True),
        use_l2warp=config_dict.get("use_l2warp", False),
        vocab_size=config_dict.get("vocab_size", 32000)
    )
    return TransformerModel(config=config)

if __name__ == "__main__":
    # Initialize the RWKV model
    # model = RWKV7Model(
    #     config=RWKV7Config(
    #         hidden_size=64,
    #         num_hidden_layers=3,
    #         head_dim=64,
    #         decay_low_rank_dim=64,
    #         gate_low_rank_dim=128,
    #         a_low_rank_dim=64,
    #         v_low_rank_dim=16,
    #         max_position_embeddings=None,
    #         vocab_size=5,
    #     )
    # )

    # model = build_loop_model({
    #     "hidden_size": 64,
    #     "num_hidden_layers": 3,
    #     "head_dim": 64,
    #     "decay_low_rank_dim": 64,
    #     "gate_low_rank_dim": 128,
    #     "a_low_rank_dim": 64,
    #     "v_low_rank_dim": 16,
    #     "max_position_embeddings": None,
    #     "vocab_size": 5,
    #     "num_block_groups": 1,
    #     "num_layers_per_group": 3,
    #     "loop_strategy": 'uniform',
    #     "loop_times": 2,  # or a dictionary like {"group_idx": 2} for custom strategy
    #     "loop_injection": 'linear_norm'
    # })
    # print("RWKV model initialized successfully.")
    # print("Model configuration:", model.config)
    # print("Model:", model)
    # print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")
    # ###test input forward
    # xs = torch.randn(64, 11, 5)  # Batch size 64, sequence length 11, hidden size 64
    # ys = torch.randn(64, 11)  # Batch size
    # # 64, sequence length 11, hidden size 64
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # xs = xs.to(device)
    # ys = ys.to(device)
    # model = model.to(device)
    # output = model(xs, ys)
    # print("Output shape:", output[0].shape)  # Should be (64,
    # # 11) for the prediction on xs
    # print("Output prediction on xs:", output[0])  # Print the prediction on xs
    model = build_transformer_model({
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "intermediate_size": 256,
        "hidden_act": "swish",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "vocab_size": 5
    })
    print("Transformer model initialized successfully.")
    print("Model configuration:", model.config)
    print("Model:", model)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")
    ###test input forward
    xs = torch.randn(64, 11, 5)  # Batch size 64, sequence length 11, hidden size 5
    ys = torch.randn(64, 11)  # Batch size 64, sequence
    # length 11, hidden size 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = xs.to(device)
    ys = ys.to(device)
    model = model.to(device)
    output = model(xs, ys)
    print("Output shape:", output[0].shape)  # Should be (64,
    # 11) for the prediction on xs
    print("Output prediction on xs:", output[0])  # Print the prediction on xs
    # print("Output:", output)  # Print the full output structure

