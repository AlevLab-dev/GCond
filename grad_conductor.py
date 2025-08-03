# gcond/grad_conductor.py
from __future__ import annotations

import contextlib
import math
from typing import Callable, Dict, List, Literal, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    # PyTorch ≥ 2.0 (2.2+ recommended)
    from torch.func import functional_call
except ImportError:
    functional_call = None  # type: ignore

# --- Type Aliases ---
GradList = List[torch.Tensor]
DataProvider = Callable[[], Tuple[torch.Tensor, torch.Tensor]]

# --- Helper Components ---
def _named_trainable_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Returns a dictionary {name: param} of trainable parameters."""
    return {k: v for k, v in module.named_parameters() if v.requires_grad}


class _FreezeContext(contextlib.AbstractContextManager):
    """A context manager to temporarily set BatchNorm layers to eval() mode."""
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self._original_training: Dict[nn.Module, bool] = {}

    def __enter__(self):
        for m in self.module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self._original_training[m] = m.training
                m.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m, was_training in self._original_training.items():
            m.train(was_training)
        return False  # Do not suppress exceptions


# --- Main Conductor Class ---
class GradientConductor:
    r"""
    Constructs a unified gradient from a set of loss functions using orthogonal
    projection and momentum smoothing. It does not modify the **model** weights
    directly but writes the result to `p.grad`, thus integrating with any
    external optimizer or GradScaler.

    Key Features
    ------------
    * No `state_dict()` copying; gradients are computed functionally.
    * Correctly handles AMP unscaling, BatchNorm freezing, and DDP sync.
    * Faster and more memory-efficient, ensuring correct operation with
        large `accumulation_steps` and multiple GPUs.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        lambdas: Dict[str, float],
        accumulation_steps: int,
        *,
        projection_max_iters: Optional[int] = None,
        norm_cap: Optional[float] = None,
        momentum_beta: float = 0.85,
        use_lion: bool = True,              # Enables Lion-style update
        trust_ratio_coef: float = 0.004,    # LARS/LAMB trust ratio coefficient
        dominance_window: int = 3,
        conflict_thresholds: Tuple[float, float, float] = (-0.75, -0.5, 0), # (critical, main, weak) thresholds
        norm_ema_beta: float = 0.95,
        tie_breaking_weights: Tuple[float, float] = (0.8, 0.2), # (stability, strength) - Tie-breaking weights
        return_raw_grad: bool = False,
        ddp_sync: Literal["avg", "broadcast", "none"] = "avg",
        freeze_bn: bool = True,
        eps: float = 1e-8,
    ) -> None:
        # --- Arguments ---
        if loss_fns.keys() != lambdas.keys():
            raise ValueError("Keys of loss_fns and lambdas must match.")
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        if functional_call is None:
            raise RuntimeError(
                "GradientConductor now requires PyTorch ≥ 2.0 with torch.func"
            )

        self.model = model
        self.loss_fns = loss_fns
        self.lambdas = {k: float(v) for k, v in lambdas.items()}
        self.acc_steps = accumulation_steps

        #self.projection_mode = projection_mode
        self.max_iters = projection_max_iters
        self.norm_cap = norm_cap

        if not (0.0 <= momentum_beta < 1.0):
            raise ValueError("momentum_beta must be in [0, 1).")

        self.beta = float(momentum_beta)
        self.use_lion = bool(use_lion)
        # Stores the scaled update for Lion/SGD
        self._out_update: Optional[GradList] = None
        if trust_ratio_coef <= 0:
            raise ValueError("trust_ratio_coef must be > 0")
        self.trust_ratio_coef = float(trust_ratio_coef)
    
        if not (0.0 <= norm_ema_beta < 1.0):
            raise ValueError("norm_ema_beta must be in [0, 1).")
        self.norm_ema_beta = norm_ema_beta

        if not (0 <= dominance_window):
            raise ValueError("dominance_window must be non-negative.")
        self.dominance_window = dominance_window
        
        # --- Validate conflict_thresholds ---
        if len(self.conflict_thresholds) != 3:
            raise ValueError("conflict_thresholds must be a tuple of 3 floats.")

        crit_thresh, main_thresh, weak_thresh = self.conflict_thresholds
        if not (crit_thresh <= main_thresh <= weak_thresh):
            raise ValueError(
                "Conflict thresholds must be in non-decreasing order "
                "(critical <= main <= weak)."
            )

        for i, th in enumerate(self.conflict_thresholds):
            if not (-1.0 <= th <= 1.0):
                raise ValueError(
                    f"Threshold at index {i} ({th}) is outside the valid "
                    f"cosine similarity range of [-1.0, 1.0]."
                )
        self.conflict_thresholds = conflict_thresholds
        
        if not (len(tie_breaking_weights) == 2 and sum(tie_breaking_weights) > 0):
            raise ValueError("tie_breaking_weights must be a tuple of 2 non-negative numbers with a positive sum.")
        self.tie_breaking_weights = tie_breaking_weights

        self.return_raw_grad = bool(return_raw_grad)

        self.ddp_sync_mode = ddp_sync
        self.freeze_bn_flag = freeze_bn
        self.eps = float(eps)

        self.is_ddp = isinstance(model, DDP)
        # A flag to check if the current process is the main one (rank 0 in DDP)
        # Used to ensure certain operations (like logging) run only once.
        self.rank0 = (
            (not self.is_ddp)
            or (torch.distributed.get_rank() == 0)  # type: ignore[attr-defined]
        )

        # --- Parameter & State Buffers ---
        self.grad_params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        if not self.grad_params:
            raise ValueError("Model has no trainable parameters.")
        self.device = self.grad_params[0].device

        self.accumulators: Dict[str, GradList] = {
            k: [torch.zeros_like(p, dtype=torch.float32) for p in self.grad_params]
            for k in self.loss_fns
        }
        self.momentum: GradList = [
            torch.zeros_like(p, dtype=torch.float32) for p in self.grad_params
        ]
        # Buffer for the projected gradient before the momentum update
        self._last_safe: Optional[GradList] = None

        # --- State for the Adaptive Arbitrator ---
        # History of "winners" to track dominance
        self.projection_history: deque[str] = deque(maxlen=self.dominance_window)
        # Previous gradients for stability calculation (in float16 to save memory)
        self.prev_accumulators: Dict[str, Optional[GradList]] = {
            k: None for k in self.loss_fns
        }
        self.norm_moving_averages: Dict[str, float] = { 
            k: 1.0 for k in self.loss_fns                   # Initialize with 1.0 for stability
        }
        self._step_idx: int = 0  # for momentum bias-correction

    # --- Metrics ---
    @staticmethod
    def _dot(g1: GradList, g2: GradList) -> torch.Tensor:
        """Computes the dot product in full precision (float32)."""
        acc = torch.zeros((), device=g1[0].device, dtype=torch.float32)
        for a, b in zip(g1, g2):
            acc += torch.sum(a.float() * b.float())
        return acc

    @staticmethod
    def _norm_sq(g: GradList) -> torch.Tensor:
        """Computes the squared L2 norm in full precision (float32)."""
        acc = torch.zeros((), device=g[0].device, dtype=torch.float32)
        for t in g:
            acc += torch.sum(t.float().pow(2))
        return acc

    # --- Core Logic ---
    def _accumulate_for_loss(
        self,
        key: str,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        params = _named_trainable_params(
            self.model.module if self.is_ddp else self.model
        )
        buffers = dict(
            (self.model.module if self.is_ddp else self.model).named_buffers()
        )
        param_names = list(params.keys())
        # Create new leaf tensors for the gradient computation. This is a key
        # pattern for `functional_call` that prevents modifying the original
        # model parameters or their gradients during this forward pass.
        param_tensors = [p.detach().requires_grad_() for p in params.values()]

        with torch.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda"),
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ):
            def _forward(*flat_params: torch.Tensor) -> torch.Tensor:
                param_dict = dict(zip(param_names, flat_params))
                out = functional_call(
                    self.model.module if self.is_ddp else self.model,
                    (param_dict, buffers),
                    (x,),
                )
                loss = (
                    self.loss_fns[key](out, y)
                    * self.lambdas[key]
                    / self.acc_steps
                )
                return loss

            loss = _forward(*param_tensors)

        # Gradients are computed with torch.autograd.grad to avoid implicit DDP sync
        grads = torch.autograd.grad(
            loss,
            param_tensors,
            retain_graph=False,
            allow_unused=True,
        )

        for t_acc, g in zip(self.accumulators[key], grads):
            if g is not None:
                t_acc.add_(g.detach().to(torch.float32))
        return loss.item()

    def _resolve_conflict(
        self,
        name_i: str,
        name_j: str,
        grads: Dict[str, GradList],
        cosine_similarity: torch.Tensor, # Accepts pre-computed cosine similarity
        norm_emas: Dict[str, float],
    ) -> None:
        """
        Resolves a conflict between gradients using a 4-zone strategy
        based on their cosine similarity.
        """
        g_i, g_j = grads[name_i], grads[name_j]
        norm_i_sq = self._norm_sq(g_i)
        norm_j_sq = self._norm_sq(g_j)
        dot_product = cosine_similarity * torch.sqrt(norm_i_sq * norm_j_sq)

        crit_thresh, main_thresh, weak_thresh = self.conflict_thresholds
        
        # --- Zone 4: CRITICAL CONFLICT (cos < crit_thresh) -> Winner-takes-all ---
        if cosine_similarity < crit_thresh:
            winner, loser = self._run_arbitrator(name_i, name_j, g_i, g_j, norm_emas)
            g_loser = grads[loser]
            # Completely zero out the loser's gradient
            for p_l in g_loser:
                p_l.zero_()
            if self.dominance_window > 0:
                self.projection_history.append(winner)

        # --- Zone 3: MAIN CONFLICT (crit_thresh <= cos < main_thresh) -> Arbitrator + PCGrad ---
        elif cosine_similarity < main_thresh:
            winner, loser = self._run_arbitrator(name_i, name_j, g_i, g_j, norm_emas)
            g_winner, g_loser = grads[winner], grads[loser]
            
            # Project the loser's gradient to be orthogonal to the winner's (PCGrad logic)
            norm_sq_winner = self._norm_sq(g_winner)
            proj_scalar = self._dot(g_loser, g_winner) / (norm_sq_winner + self.eps)
            for p_l, p_w in zip(g_loser, g_winner):
                p_l.add_(p_w, alpha=-proj_scalar)
            
            if self.dominance_window > 0:
                self.projection_history.append(winner)

        # --- Zone 2: WEAK CONFLICT (main_thresh <= cos < weak_thresh) -> Symmetric Projection ---
        elif cosine_similarity < weak_thresh:
            proj_scalar_i_on_j = dot_product / (norm_j_sq + self.eps)
            proj_scalar_j_on_i = dot_product / (norm_i_sq + self.eps)
            
            # Apply updates simultaneously to avoid sequential dependency
            for p_i, p_j in zip(g_i, g_j):
                update_for_pi = p_j.mul(-proj_scalar_i_on_j)
                p_j.add_(p_i, alpha=-proj_scalar_j_on_i)    # p_j uses the original p_i
                p_i.add_(update_for_pi)                     # p_i uses the pre-calculated update

        # --- Zone 1: AGREEMENT (cos >= weak_thresh) -> No action taken ---
        # The loop in _project() will terminate if this is the most severe
        # conflict found. This `else` block is a placeholder for clarity.
        else:
            pass

    def _run_arbitrator(
        self,
        name_i: str,
        name_j: str,
        g_i: GradList,
        g_j: GradList,
        norm_emas: Dict[str, float],
    ) -> Tuple[str, str]:
        """The Arbitrator logic, separated for clarity, to select a winner/loser."""
        winner, loser = name_i, name_j # Initial assumption

        # 1. Dominance Check: if one task has won consistently, it wins automatically.
        is_i_dominant = (self.dominance_window > 0 and
                        len(self.projection_history) == self.dominance_window and
                        self.projection_history.count(name_i) == self.dominance_window)
        is_j_dominant = (self.dominance_window > 0 and
                        len(self.projection_history) == self.dominance_window and
                        self.projection_history.count(name_j) == self.dominance_window)
        
        if is_i_dominant and not is_j_dominant:
            return name_i, name_j
        if is_j_dominant and not is_i_dominant:
            return name_j, name_i

        # 2. Tie-Breaking via Hybrid Score (if no dominance is established)
        norm_i = self._norm_sq(g_i).sqrt()
        norm_j = self._norm_sq(g_j).sqrt()

        # 2a. Stability Score (cosine similarity with its own previous gradient)
        stability_i = torch.tensor(0.0, device=self.device)
        if self.prev_accumulators.get(name_i):
            dot_prev = self._dot(g_i, self.prev_accumulators[name_i])
            norm_prev_i = self._norm_sq(self.prev_accumulators[name_i]).sqrt()
            stability_i = dot_prev / (norm_i * norm_prev_i + self.eps)

        stability_j = torch.tensor(0.0, device=self.device)
        if self.prev_accumulators.get(name_j):
            dot_prev = self._dot(g_j, self.prev_accumulators[name_j])
            norm_prev_j = self._norm_sq(self.prev_accumulators[name_j]).sqrt()
            stability_j = dot_prev / (norm_j * norm_prev_j + self.eps)

        # 2b. Strength Score (gradient norm relative to its moving average)
        scaled_norm_i = norm_i / (norm_emas[name_i] + self.eps)
        scaled_norm_j = norm_j / (norm_emas[name_j] + self.eps)
        total_scaled_norm = scaled_norm_i + scaled_norm_j
        strength_i = scaled_norm_i / (total_scaled_norm + self.eps)
        strength_j = scaled_norm_j / (total_scaled_norm + self.eps)

        # 2c. Final weighted score
        w_stability, w_strength = self.tie_breaking_weights
        # Ignore negative stability (when a gradient flips direction)
        score_i = w_stability * stability_i.clamp(min=0) + w_strength * strength_i
        score_j = w_stability * stability_j.clamp(min=0) + w_strength * strength_j

        if score_j > score_i:
            winner, loser = name_j, name_i
        
        return winner, loser

    def _project(self) -> Tuple[GradList, Dict[str, float]]:
        """
        Iteratively finds and resolves conflicts between gradients until all
        cosine similarities are non-negative or the iteration limit is reached.
        """
        stats = {"proj_iters": 0.0, "min_cosine_sim": 1.0} # 1.0 indicates perfect agreement
        if len(self.loss_fns) <= 1:
            key = next(iter(self.loss_fns.keys()))
            grad_list = [g.clone() for g in self.accumulators[key]]
            stats[f'raw_norm/{key}'] = self._norm_sq(grad_list).sqrt().item()
            return grad_list, stats

        grads = {k: [t.clone() for t in v] for k, v in self.accumulators.items()}
        for name, grad_list in grads.items():
            raw_norm = self._norm_sq(grad_list).sqrt()
            stats[f'raw_norm/{name}'] = raw_norm.item()
            
            current_ema = self.norm_moving_averages[name]
            new_ema = self.norm_ema_beta * current_ema + (1.0 - self.norm_ema_beta) * raw_norm.item()
            self.norm_moving_averages[name] = max(new_ema, self.eps)
            stats[f'norm_ema/{name}'] = self.norm_moving_averages[name]
        
        if self.norm_cap is not None:
            for grad_list in grads.values():
                total_norm = torch.sqrt(self._norm_sq(grad_list))
                if total_norm > self.norm_cap:
                    clip_coef = self.norm_cap / (total_norm + self.eps)
                    for t in grad_list:
                        t.mul_(clip_coef)

        grad_names = list(grads.keys())
        crit_thresh, main_thresh, weak_thresh = self.conflict_thresholds
        
        # A reasonable default for max_iters to prevent potential infinite loops.
        for iter_num in range(self.max_iters or len(grad_names) * 2):
            # Start the search for the most conflicting pair. Any pair with a cosine
            # similarity above `weak_thresh` will be ignored.
            min_cosine_sim = torch.tensor(weak_thresh, device=self.device)
            conflict_pair = None
            
            # --- Find the pair with the minimum cosine similarity ---
            for i in range(len(grad_names)):
                for j in range(i + 1, len(grad_names)):
                    g_i, g_j = grads[grad_names[i]], grads[grad_names[j]]
                    norm_i_sq = self._norm_sq(g_i)
                    norm_j_sq = self._norm_sq(g_j)
                    
                    # Skip pairs with zero-norm gradients
                    if norm_i_sq < self.eps or norm_j_sq < self.eps:
                        continue
                        
                    dot = self._dot(g_i, g_j)
                    cosine = dot / (torch.sqrt(norm_i_sq * norm_j_sq) + self.eps)
                    
                    if cosine < min_cosine_sim:
                        min_cosine_sim = cosine
                        conflict_pair = (grad_names[i], grad_names[j])
            
            # --- If the most acute angle is already in the agreement zone, we can exit early ---
            if conflict_pair is None:
                break
            
            if iter_num == 0:
                stats['min_cosine_sim'] = min_cosine_sim.item()
            stats['proj_iters'] = float(iter_num + 1)

            # Resolve the most significant conflict found in this iteration
            self._resolve_conflict(
                conflict_pair[0], 
                conflict_pair[1], 
                grads, 
                min_cosine_sim,  # Pass the pre-computed cosine
                self.norm_moving_averages
            )

        # Final aggregation of resolved gradients
        final_grads = [torch.zeros_like(p, dtype=torch.float32) for p in self.grad_params]
        for grad_list in grads.values():
            for i, grad_tensor in enumerate(grad_list):
                final_grads[i].add_(grad_tensor)

        return final_grads, stats

    def _momentum_update(self, g: GradList) -> None:
        """Applies momentum update (EMA) with optional Lion and Trust-Ratio scaling."""
        self._step_idx += 1
        b = self.beta
        one_minus_b = 1.0 - b
        scaled = []
        with torch.no_grad():
            for m, g_, p in zip(self.momentum, g, self.grad_params):
                # 1. EMA
                m.mul_(b).add_(g_, alpha=one_minus_b)

                # 2. Get the update direction from the updated momentum buffer m.
                #    (sign for Lion, the momentum vector itself for SGD-style updates)
                upd = torch.sign(m) if self.use_lion else m

                # 3. Trust-Ratio
                upd_norm = upd.norm()
                if upd_norm < 1e-12:     # Guard against division by zero
                    trust = 1.0
                else:
                    trust = (p.detach().norm() /
                             (upd_norm + self.eps)) * self.trust_ratio_coef
                scaled.append(upd * trust)
        # Store the final, scaled update vector to be used by _write_to_model
        self._out_update = scaled

    def _write_to_model(self) -> None:
        # Determine the source gradients to write to the model
        if self.return_raw_grad:
            # Use the projected gradients directly without momentum
            src = self._last_safe
        else:
            # Lion update already contains the scaled sign(m)*trust; no bias-correction needed
            # For SGD-style momentum, apply bias-correction
            src = (self._out_update if self.use_lion else
                   [m / (1.0 - self.beta ** self._step_idx) for m in self.momentum])
        for p, g in zip(self.grad_params, src):
            # .detach() ensures no lingering graph dependencies
            p.grad = g.to(dtype=p.dtype).detach()

    # --- Public API ---
    def step(
        self,
        data_provider: DataProvider,
    ) -> Dict[str, float]:
        """
        Computes and writes the final gradient to model.parameters().
        Returns a dictionary of statistics for logging.
        """

        # 0) Zero-out gradient accumulators
        with torch.no_grad():
            for v in self.accumulators.values():
                for t in v:
                    t.zero_()
        accumulated_losses = {k: 0.0 for k in self.loss_fns}
        # 1) Accumulate gradients for each loss function
        try:
            # Temporarily switch model to eval() mode for deterministic forward passes.
            # This disables dropout and ensures BatchNorm uses running statistics,
            # which is crucial for consistent gradient calculations across all
            # accumulation steps.
            _orig_training: Optional[bool] = None
            if self.freeze_bn_flag:
                _orig_training = self.model.training
                self.model.eval()

            with torch.enable_grad():
                for _ in range(self.acc_steps):
                    x, y = data_provider()
                    # x and y are assumed to be on the correct device, as handled
                    # by the data_provider

                    for key in self.loss_fns:
                        loss_value = self._accumulate_for_loss(key, x, y)
                        accumulated_losses[key] += loss_value
        finally:
            if self.freeze_bn_flag:
                # Restore the model's original training mode
                self.model.train(_orig_training)  # type: ignore[arg-type]
        # 2) The rest can be done without gradient tracking
        with torch.no_grad():
            # Store the raw gradients BEFORE projection for the next step's stability calculation
            for name, acc_list in self.accumulators.items():
                dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float16
                self.prev_accumulators[name] = [p.clone().to(dtype) for p in acc_list]

            safe_grad, proj_stats = self._project()
            self._last_safe = safe_grad
            self._momentum_update(safe_grad)

            # --- DDP Synchronization (if enabled) ---
            if self.is_ddp and self.ddp_sync_mode != "none":
                # Sync the exact tensors that will be written to p.grad
                sync_tensors = (
                    self._last_safe
                    if self.return_raw_grad
                    else (self._out_update if self.use_lion else self.momentum)
                )
                for t in sync_tensors:
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)  # type: ignore[attr-defined]
                    if self.ddp_sync_mode == "avg":
                        t.div_(torch.distributed.get_world_size())  # type: ignore[attr-defined]

            self._write_to_model()

            # Calculate the norm of the actual gradient being applied
            if self.return_raw_grad:
                _g_for_norm = self._last_safe
            elif self.use_lion:
                _g_for_norm = self._out_update  # type: ignore[arg-type]
            else:
                _g_for_norm = self.momentum

            final_grad_norm = math.sqrt(self._norm_sq(_g_for_norm).item())
            
            # Aggregate all statistics into a single dictionary
            final_stats = {
                "final_grad_norm": final_grad_norm,
                "step": self._step_idx
            }
            # Add projection stats
            final_stats.update(proj_stats)

            for key, value in accumulated_losses.items():
                final_stats[f'loss/{key}'] = value
            
            return final_stats
