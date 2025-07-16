## LUMIR 2024

This file contains configuration details for experiments conducted for the **LUMIR 2024** challenge.

---

### 🔧 Loss Function

The loss function used in this work is defined as:  
	Loss = 1 - NCC + λ × Diffusion Regularization

- **NCC**: Normalized Cross-Correlation  
- **λ **: Regularization weight, set to **1** (same as in all baseline methods)

**NCC implementation used**:  
[yihao6/vfa – ncc_loss.py (line 8)](https://github.com/yihao6/vfa/blob/07d933d8396d5f9f8d7e3dac74d8c95b3f8e8314/vfa/losses/ncc_loss.py#L8)

> ⚠️ Note: The NCC implementations from **VoxelMorph** and **TransMorph** did not perform as well in our setup, potentially due to PyTorch version differences.

---

### 📦 Batch Size

- `batch_size = 1`

---

### 🧪 Training Configuration

#### Epochs
- Total epochs: **1500**
- Each epoch processes **100 fixed-moving pairs** (this is arbitrary just to maintain consistent comparison across methods)

#### Learning Rate Strategies Tested

1. **Constant**: `1e-4`
2. **Polynomial decay**: From `1e-3` to `1e-5`
3. **Cosine Annealing with Warm Restarts**:
   - LR cycles between `1e-3` and `1e-5`
   - Restarts at epochs: **100**, **300** (100 + 200), **700** (100 + 200 + 400)

✅ The **cosine annealing with warm restarts** strategy yielded the best performance in our experiments.

---

### 💾 Model Checkpointing

- The best model is selected based on the **lowest `1 - NCC`** on the validation set.
