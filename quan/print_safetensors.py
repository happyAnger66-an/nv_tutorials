from safetensors import safe_open

import sys

if __name__ == "__main__":
    with safe_open(sys.argv[1], framework='pt') as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            print(f'key {k} shape {tensor.shape} {tensor.dtype}')
        

