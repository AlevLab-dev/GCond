# GradientConductor

`GradientConductor` is an advanced gradient management utility for PyTorch, designed for training models with multiple, potentially conflicting, loss functions. It serves as an intelligent alternative to standard gradient accumulation, especially effective in scenarios with large effective batch sizes where naive gradient summing can become statistically significant and suboptimal.

The core idea is to analyze and resolve conflicts between gradients from different loss sources *before* they are applied, leading to more stable and effective training. It calculates a unified gradient and writes it directly to `model.parameters().grad`, making it fully compatible with any standard PyTorch optimizer (`AdamW`, `SGD`, etc.) and `torch.amp.GradScaler`.

## Key Features

  * **Conflict-Aware Gradient Merging**: Instead of simply summing gradients, it uses cosine similarity to detect and resolve conflicts between different loss gradients. It employs a multi-zone strategy, from symmetric projection for weak conflicts to a winner-takes-all approach for critical ones.
  * **Adaptive Arbitrator**: When gradients are in direct opposition, an arbitrator decides the "winner" based on a hybrid score of training stability and relative gradient strength.
  * **Integrated Momentum & Trust-Ratio**: It manages its own momentum update, optionally replacing the optimizer's first moment (`beta1` in Adam). It can also apply a trust-ratio scaling (similar to LARS/LAMB) for more stable updates.
  * **Optimizer-Agnostic**: Works seamlessly with any external optimizer. You compute the gradient with `Conductor`, and the optimizer just applies it.
  * **Efficient Implementation**: Built on `torch.func.functional_call` for memory and speed efficiency, avoiding expensive `state_dict` copying and correctly handling batch normalization during gradient computation.

## Installation

At the moment, you can simply clone this repository and copy the `gcond` directory into your project.

```bash
git clone <your-repo-url>
cp -r <your-repo-url>/gcond /path/to/your/project/
```

## How to Use

Using `GradientConductor` involves replacing the standard `loss.backward()` loop with a call to `conductor.step()`.

### 1\. Define your loss functions

Your model should return the outputs needed for each loss. The loss functions themselves will be passed to the `GradientConductor`.

```python
# In your model...
def calculate_losses(self, pred, target):
    l1_loss = F.l1_loss(pred, target)
    ssim_loss = 1.0 - self.ssim_module(pred, target)
    return {"l1": l1_loss, "ssim": ssim_loss}

# In your training script, define the callables
def l1_loss_fn(model_output, target):
    # This function will be called by the conductor
    # In this example, model_output is a tuple (pred_patches, mask, etc.)
    # and we assume the logic is handled inside the model.
    # The key is to map the model's forward pass to the loss calculation.
    pred_patches, _, mask = model_output 
    # ... calculate l1 loss ...
    return l1_loss

def ssim_loss_fn(model_output, target):
    # ... calculate ssim loss ...
    return ssim_loss

loss_fns = {'l1': l1_loss_fn, 'ssim': ssim_loss_fn}
lambdas = {'l1': 0.85, 'ssim': 0.15}
```

### 2\. Initialize `GradientConductor` and Optimizer

The `Conductor` takes over the momentum calculation. It's recommended to set your optimizer's `beta1` to `0.0` to avoid redundant momentum tracking.

```python
from gcond.grad_conductor import GradientConductor
import torch.optim as optim

# ... model setup ...

# Note: betas=(0.0, 0.95). The first beta is handled by the Conductor.
optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.0, 0.95))

conductor = GradientConductor(
    model=model,
    loss_fns=loss_fns,
    lambdas=lambdas,
    accumulation_steps=24,
    momentum_beta=0.9 # This acts as the new beta1 for the optimizer
)
```

### 3\. Create a Data Provider

The conductor pulls data through a simple `data_provider` function that you define. This function should handle data loading, moving to the device, and any GPU-based augmentations. It must return a tuple of `(input, target)`.

```python
# Create an iterator from your DataLoader
train_iter = iter(train_loader)

def data_provider():
    # Gets one batch, moves to device, applies augmentations
    images, _ = next(train_iter)
    images = images.to(device)
    augmented_images = augmenter(images)
    # For a self-supervised task like MAE, input and target are the same
    return augmented_images, augmented_images
```

### 4\. Update Your Training Step

Replace your `loss.backward()` and `scaler` calls with a single call to `conductor.step()`.

```python
# --- Old way ---
# optimizer.zero_grad()
# for _ in range(ACCUMULATION_STEPS):
#     images, _ = next(data_iterator)
#     with autocast():
#         loss = model(images)
#         loss = loss / ACCUMULATION_STEPS
#     scaler.scale(loss).backward()
#
# scaler.unscale_(optimizer)
# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# scaler.step(optimizer)
# scaler.update()

# --- New way with GradientConductor ---
optimizer.zero_grad()

# The conductor handles the accumulation loop, loss calculation,
# gradient projection, and momentum update internally.
# It populates `p.grad` for each parameter.
stats = conductor.step(data_provider=data_provider)

# Optional: You can still apply global gradient clipping if needed.
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# The optimizer's job is now just to apply the final gradients.
optimizer.step()

# Log useful stats from the conductor
print(f"Final Grad Norm: {stats['final_grad_norm']:.4f}, Cosine Similarity: {stats['min_cosine_sim']:.4f}")
```

## Configuration Parameters

Here is a brief overview of the key parameters for the `GradientConductor` class.

### Core Parameters
* `model` (`nn.Module`): The model you are training.
* `loss_fns` (`Dict[str, Callable]`): A dictionary mapping loss names to their corresponding loss functions.
* `lambdas` (`Dict[str, float]`): A dictionary mapping loss names to their weight (lambda) values.
* `accumulation_steps` (`int`): The number of batches to accumulate gradients over before performing an update.

### Conflict Resolution
* `projection_max_iters` (`Optional[int]`): The maximum number of iterations to resolve gradient conflicts. Defaults to a reasonable value if not set.
* `conflict_thresholds` (`Tuple[float, float, float]`): A tuple of three cosine similarity values `(critical, main, weak)` that define the conflict zones. Must be in non-decreasing order (e.g., `(-0.75, -0.5, 0.0)`).
* `dominance_window` (`int`): The number of recent steps to check for task dominance. A task that "wins" conflicts for this many steps in a row will automatically win the next conflict. Set to `0` to disable.
* `tie_breaking_weights` (`Tuple[float, float]`): The weights `(stability, strength)` used by the arbitrator to break ties when no task is dominant.

### Gradient Normalization and Updates
* `norm_cap` (`Optional[float]`): An optional value to cap the L2 norm of each raw gradient *before* conflict resolution.
* `momentum_beta` (`float`): The momentum coefficient (like `beta1` in Adam). This replaces the optimizer's momentum.
* `use_lion` (`bool`): If `True`, uses a Lion-style update (`sign(momentum)`). If `False`, uses a standard momentum update.
* `trust_ratio_coef` (`float`): The coefficient for trust-ratio scaling, which adapts the update size based on the ratio of parameter norm to update norm.

### Technical Parameters
* `return_raw_grad` (`bool`): If `True`, the final gradient written to `p.grad` will be the projected gradient *without* the momentum update.
* `freeze_bn` (`bool`): If `True`, automatically sets BatchNorm layers to `eval()` mode during gradient accumulation for deterministic behavior.
* `eps` (`float`): A small epsilon value to prevent division by zero in normalization calculations.