# GCond/grad_conductor.py
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
        accumulation_steps: int, # should be divisible by the number of loss functions
        *,
        projection_max_iters: Optional[int] = None,
        norm_cap: Optional[float] = None,
        momentum_beta: float = 0.85,
        use_lion: bool = False,              # Enables Lion-style update
        trust_ratio_coef: float = 1e-4,    # LARS/LAMB trust ratio coefficient
        trust_ratio_clip: float = 100.0,
        dominance_window: int = 3,
        conflict_thresholds: Tuple[float, float, float] = (-0.8, -0.5, 0), # (critical, main, weak) thresholds
        norm_ema_beta: float = 0.95,
        tie_breaking_weights: Tuple[float, float] = (0.8, 0.2), # (stability, strength) - Tie-breaking weights
        return_raw_grad: bool = False,
        remap_power: float = 2.0,           # Power for non-linear angle remapping
        use_smooth_logic: bool = True,      # Use the new smooth conflict resolution
        stochastic_accumulation: bool = True, # or sequential loss calculation
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

        self.max_iters = projection_max_iters
        self.norm_cap = norm_cap

        if not (0.0 <= momentum_beta < 1.0):
            raise ValueError("momentum_beta must be in [0, 1).")

        self.beta = float(momentum_beta)
        self.use_lion = bool(use_lion)
        
        if trust_ratio_coef <= 0:
            raise ValueError("trust_ratio_coef must be > 0")
        self.trust_ratio_coef = float(trust_ratio_coef)
        self.trust_ratio_clip = trust_ratio_clip

        if remap_power <= 0:
            raise ValueError("remap_power must be positive.")
        self.remap_power = float(remap_power)
        self.use_smooth_logic = bool(use_smooth_logic)
        self.stochastic_accumulation = bool(stochastic_accumulation)

    
        if not (0.0 <= norm_ema_beta < 1.0):
            raise ValueError("norm_ema_beta must be in [0, 1).")
        self.norm_ema_beta = norm_ema_beta

        if not (0 <= dominance_window):
            raise ValueError("dominance_window must be non-negative.")
        self.dominance_window = dominance_window
        
        self.conflict_thresholds = conflict_thresholds
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
            k: [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.grad_params]
            for k in self.loss_fns
        }
        self.momentum: GradList = [
            torch.zeros_like(p, dtype=torch.bfloat16) for p in self.grad_params
        ]
        # Buffer for the final gradient update after momentum and trust-ratio scaling
        self.final_update: GradList = [
            torch.zeros_like(p, dtype=torch.bfloat16) for p in self.grad_params
        ]
        # Buffer for the projected gradient before the momentum update
        self._last_safe: Optional[GradList] = None

        # --- State for the Adaptive Arbitrator ---
        # History of "winners" to track dominance
        self.projection_history: deque[str] = deque(maxlen=self.dominance_window)

        # Previous gradients for stability calculation (in float16 to save memory)
        self.prev_accumulators: Dict[str, GradList] = {
            k: [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.grad_params]
            for k in self.loss_fns
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

    def _get_effective_alpha(
        self,
        cosine_sim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the 'effective' conflict angle using a non-linear remapping
        based on cosine similarity thresholds.
        """
        c = cosine_sim.clamp(-1.0, 1.0)
        t_crit, t_main, t_weak = self.conflict_thresholds

        pi_half = torch.tensor(torch.pi / 2.0, device=c.device)

        if c >= t_main:
            # Normalize c from its actual range [t_main, t_weak] to t in [0, 1]
            t = (c - t_main) / (t_weak - t_main + self.eps)
            t = t.clamp(0.0, 1.0)
            # Map t [0, 1] to angle [pi/2, 0]
            return pi_half * (1.0 - t)
        elif c > t_crit:
            # Stretches cosine [-0.75, 0] to angle [pi, pi/2] with non-linearity
            # 1. Normalize c to t in [0, 1] for this interval
            t = (c - t_main) / (t_crit - t_main + self.eps)
            # 2. Apply power for non-linearity
            t_powered = t.pow(self.remap_power)
            # 3. Interpolate angle
            return pi_half + pi_half * t_powered
        else: # c <= t_crit
            return torch.tensor(torch.pi, device=c.device)

    # --- Core Logic ---
    def _accumulate_for_loss(
        self,
        key: str,
        x: torch.Tensor,
        y: torch.Tensor,
        normalization_factor: float,
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
                    / normalization_factor  
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
                t_acc.add_(g.detach().to(torch.bfloat16))
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
            return name_j, name_i
        if is_j_dominant and not is_i_dominant:
            return name_i, name_j

        # 2. Tie-Breaking via Hybrid Score (if no dominance is established)
        norm_i = self._norm_sq(g_i).sqrt()
        norm_j = self._norm_sq(g_j).sqrt()

        # 2a. Stability Score (cosine similarity with its own previous gradient)
        dot_prev = self._dot(g_i, self.prev_accumulators[name_i])
        norm_prev_i = self._norm_sq(self.prev_accumulators[name_i]).sqrt()
        stability_i = dot_prev / (norm_i * norm_prev_i + self.eps)

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
        Resolves conflicts between gradients and returns the final unified gradient.
        Can use either the original iterative 4-zone logic or the new smooth logic.
        """
        stats = {"proj_iters": 0.0, "min_cosine_sim": 1.0} # 1.0 indicates perfect agreement
        if len(self.loss_fns) <= 1:
            key = next(iter(self.loss_fns.keys()))
            grad_list = [g.clone() for g in self.accumulators[key]]
            stats[f'raw_norm/{key}'] = self._norm_sq(grad_list).sqrt().item()
            return grad_list, stats

        grads = self.accumulators
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
        crit_thresh, _, weak_thresh = self.conflict_thresholds
        
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
                        conflict_pair = (grad_names[i], grad_names[j], dot, norm_i_sq, norm_j_sq)
            
            # If the most severe conflict found is actually an agreement, we can stop.
            if conflict_pair is None or min_cosine_sim >= weak_thresh:
                break
            
            if iter_num == 0:
                stats['min_cosine_sim'] = min_cosine_sim.item()
            stats['proj_iters'] = float(iter_num + 1)

            # Unpack the conflicting pair's data
            name_i, name_j, dot, norm_i_sq, norm_j_sq = conflict_pair

            # --- SELECT CONFLICT RESOLUTION STRATEGY ---
            if not self.use_smooth_logic:
                # --- STRATEGY 1: Original 4-zone step-logic ---
                self._resolve_conflict(
                    name_i, 
                    name_j, 
                    grads, 
                    min_cosine_sim,
                    self.norm_moving_averages
                )
            else:
                # --- STRATEGY 2: New smooth, remapped logic ---
                # 1. Determine winner/loser using the existing arbitrator
                g_i, g_j = grads[name_i], grads[name_j]
                winner_name, loser_name = self._run_arbitrator(
                    name_i, name_j, g_i, g_j, self.norm_moving_averages
                )
                if self.dominance_window > 0:
                    self.projection_history.append(winner_name)

                # 2. Assign gradients and norms according to winner/loser roles
                if winner_name == name_i:
                    g_w, g_l, n_w_sq, n_l_sq = g_i, g_j, norm_i_sq, norm_j_sq
                else:
                    g_w, g_l, n_w_sq, n_l_sq = g_j, g_i, norm_j_sq, norm_i_sq

                # 3. Get effective angle and scaling factors for modulation
                alpha_eff = self._get_effective_alpha(min_cosine_sim)
                s_w = torch.sin(alpha_eff)
                pi_half = torch.tensor(torch.pi / 2.0, device=alpha_eff.device)
                s_l = torch.sin(torch.minimum(alpha_eff, pi_half))

                # 4. Apply scaled projection update, avoiding sequential dependencies.
                # Pre-calculate final alphas for the update operations
                alpha_w = -s_w * (dot / (n_l_sq + self.eps))
                alpha_l = -s_l * (dot / (n_w_sq + self.eps))

                for p_w, p_l in zip(g_w, g_l):
                    # First, calculate both update vectors based on original tensors
                    update_for_winner = p_l.mul(alpha_w)
                    update_for_loser = p_w.mul(alpha_l)

                    # Then, apply both updates in-place
                    p_w.add_(update_for_winner)
                    p_l.add_(update_for_loser)

        # Final aggregation of resolved gradients
        grad_names = list(grads.keys())

        # If there are no gradients, return an empty list
        if not grad_names:
            return [torch.zeros_like(p) for p in self.grad_params], stats

        # Use the first gradient list as the base for summation.
        # Convert it to float32 to serve as a high-precision accumulator.
        final_grads = [g.to(torch.float32) for g in grads[grad_names[0]]]

        # Add the remaining gradient lists to the base list in-place.
        for name in grad_names[1:]:
            for final_g, task_g in zip(final_grads, grads[name]):
                final_g.add_(task_g)  # In-place addition

        return final_grads, stats

    def _momentum_update(self, g: GradList) -> None:
        """Applies momentum update (EMA) with optional Lion and Trust-Ratio scaling."""
        self._step_idx += 1
        b = self.beta
        one_minus_b = 1.0 - b

        # Adam-style bias correction factor
        bias_correction = 1.0 - b ** self._step_idx

        with torch.no_grad():
            for i, (m, g_, p) in enumerate(zip(self.momentum, g, self.grad_params)):
                # 1. Update the moving average (momentum) - no change here
                m.mul_(b).add_(g_, alpha=one_minus_b)

                if self.use_lion:
                    # --- IMPROVED LION + LARS LOGIC ---

                    # 2. Apply bias correction to the momentum. This is crucial
                    # for stabilizing direction and magnitude in early steps.
                    m_corrected = m / bias_correction
                    
                    # 3. Get update direction from the corrected momentum (Lion-style)
                    update_direction = torch.sign(m_corrected)

                    # 4. Adaptive learning rate in LARS style (CORRECTED LOGIC)
                    p_norm = p.detach().norm()
                    # Use the norm of the CORRECTED momentum, not sign(m)
                    m_corrected_norm = m_corrected.norm()

                    # Calculate the trust ratio
                    if m_corrected_norm < self.eps or p_norm < self.eps:
                        trust_ratio = torch.tensor(1.0, device=p.device)
                    else:
                        trust_ratio = p_norm / m_corrected_norm

                    trust_ratio = torch.clamp(trust_ratio, max=self.trust_ratio_clip)
                    
                    # Final update step: Direction * Adaptive LR * Base LR
                    learning_rate = trust_ratio * self.trust_ratio_coef
                    self.final_update[i] = update_direction * learning_rate
                
                else:
                    # --- STANDARD MOMENTUM (SGD-style) LOGIC ---
                    # Apply bias correction and write to final_update for unification
                    # and efficiency.
                    self.final_update[i] = m / bias_correction

    def _write_to_model(self) -> None:
        # Determine the source gradients to write to the model
        if self.return_raw_grad:
            # Use the projected gradients directly without momentum
            src = self._last_safe
        else:
            # Now, self.final_update holds the correct result for both modes
            # (Lion+LARS and standard momentum with bias correction).
            src = self.final_update

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
                if self.stochastic_accumulation:
                    loss_keys = list(self.loss_fns.keys())
                    num_losses = len(loss_keys)

                    if num_losses == 0:
                        steps_per_loss = self.acc_steps
                    else:
                        if self.acc_steps % num_losses != 0:
                            raise ValueError(
                                f"In stochastic mode, accumulation_steps ({self.acc_steps}) "
                                f"must be divisible by the number of loss fns ({num_losses})."
                            )
                        steps_per_loss = self.acc_steps // num_losses

                    for key in loss_keys:
                        if steps_per_loss == 0:
                            continue
                        for _ in range(steps_per_loss):
                            x, y = data_provider()
                            loss_value = self._accumulate_for_loss(key, x, y, float(steps_per_loss))
                            accumulated_losses[key] += loss_value
                else:
                    for _ in range(self.acc_steps):
                        x, y = data_provider()
                        for key in self.loss_fns:
                            loss_value = self._accumulate_for_loss(key, x, y, float(self.acc_steps))
                            accumulated_losses[key] += loss_value

        finally:
            if self.freeze_bn_flag:
                # Restore the model's original training mode
                self.model.train(_orig_training)  # type: ignore[arg-type]
        # 2) The rest can be done without gradient tracking
        with torch.no_grad():
            safe_grad, proj_stats = self._project()
            self._last_safe = safe_grad
            self._momentum_update(safe_grad)

            # --- DDP Synchronization (if enabled) ---
            if self.is_ddp and self.ddp_sync_mode != "none":
                sync_tensors = (
                    self._last_safe if self.return_raw_grad else self.final_update
                )
                for t in sync_tensors:
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)  # type: ignore[attr-defined]
                    if self.ddp_sync_mode == "avg":
                        t.div_(torch.distributed.get_world_size())  # type: ignore[attr-defined]

            self._write_to_model()

            # Calculate the norm of the actual gradient being applied
            _g_for_norm = self._last_safe if self.return_raw_grad else self.final_update
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

            # --- Buffer Swap for Next Iteration ---
            # Swap accumulators with prev_accumulators. This efficiently moves the raw
            # gradients from the current step into prev_accumulators for the next
            # step's stability calculation, without any memory copy.
            # The current prev_accumulators (which contains old gradients) becomes
            # the new accumulators, and will be zeroed out at the start of the next step.
            self.accumulators, self.prev_accumulators = self.prev_accumulators, self.accumulators
            
            return final_stats

    def state_dict(self) -> Dict[str, any]:
        """
        Returns a state dictionary containing all essential buffers and variables
        for checkpointing.

        The dictionary includes momentum buffers, previous step's gradients,
        norm moving averages, projection history, and the internal step counter.
        """
        return {
            "momentum": self.momentum,
            "prev_accumulators": self.prev_accumulators,
            "norm_moving_averages": self.norm_moving_averages,
            "projection_history": list(self.projection_history),
            "_step_idx": self._step_idx,
        }
    
    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        """
        Loads the conductor's state from a state dictionary.

        Args:
            state_dict (Dict[str, any]): A dictionary containing the state to load,
                typically obtained from a call to `state_dict()`.
        """
        # 1. Restore Tensors (momentum and prev_accumulators)
        # Use .copy_ to load in-place, which correctly handles device placement.
        for i, p in enumerate(state_dict["momentum"]):
            self.momentum[i].copy_(p.to(self.device))

        for key, grad_list in state_dict["prev_accumulators"].items():
            if key in self.prev_accumulators:
                for i, p in enumerate(grad_list):
                    self.prev_accumulators[key][i].copy_(p.to(self.device))

        # 2. Restore non-tensor state
        self.norm_moving_averages = state_dict["norm_moving_averages"]
        self._step_idx = state_dict["_step_idx"]

        # 3. Restore the deque with the correct maxlen from the current config
        self.projection_history = deque(
            state_dict["projection_history"], maxlen=self.dominance_window
        )

        # 4. For consistency, zero out the current accumulators, as they
        # would be at the start of a `step`.
        with torch.no_grad():
            for v in self.accumulators.values():
                for t in v:
                    t.zero_()