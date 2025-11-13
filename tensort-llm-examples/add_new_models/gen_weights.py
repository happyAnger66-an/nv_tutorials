import torch

from safetensors.torch import save_file

if __name__ == "__main__":
    tensors = {
        "transformers.vocab_embedding.weight": torch.rand((151936, 4096), dtype=torch.bfloat16),
        "lm_head.weight": torch.rand((151936, 4096), dtype=torch.half)
    }
    
    save_file(tensors, "xiaoan_test/rank0.safetensors")