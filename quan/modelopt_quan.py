import torch
from torch.utils.data import DataLoader
from torch import nn    

from torchvision import datasets
from torchvision.transforms import ToTensor

import modelopt.torch.quantization as mtq

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f"Using {device} device")

from model_def import NeuralNetwork
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
# Setup the model
#model = get_model()

# Select quantization config
config = mtq.INT8_SMOOTHQUANT_CFG

# Quantization need calibration data. Setup calibration data loader
# An example of creating a calibration data loader looks like the following:
#data_loader = get_dataloader(num_samples=calib_size)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
data_loader = DataLoader(training_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_layer_outputs(model, hook_dict):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
            hook = module.register_forward_hook(
                lambda m, i, o, name=name: hook_dict.update({name: o.detach()})
            )
            hooks.append(hook)
    return hooks

float_outputs = {}
quant_outputs = {}
current_round = 0
# Define forward_loop. Please wrap the data loader in the forward_loop
def forward_loop(model):
    for batch in data_loader:
        x, y = batch
        X = x.to(device)
        model(X)
       
print(f'before quantize\n')
test(data_loader, model, loss_fn)
# Quantize the model and perform calibration (PTQ)
q_model = mtq.quantize(model, config, forward_loop)

current_round = 1
print(f'after quantize\n')
test(data_loader, model, loss_fn)

for batch in data_loader:
    hooks_funcs = []
    hooks_funcs = get_layer_outputs(model, float_outputs)
        
    x, y = batch
    X = x.to(device)
    model(X)
    
    for hook in hooks_funcs:
        hook.remove()
    
    hooks_funcs = get_layer_outputs(model, quant_outputs)
    q_model(X)
    for hook in hooks_funcs:
        hook.remove()
        
    for layer_name in float_outputs:
        if layer_name in quant_outputs:
            float_val = float_outputs[layer_name].flatten().numpy()
            quant_val = quant_outputs[layer_name].flatten().numpy()
            print(f'layer_name:{layer_name}')
            print(f'float_val {float_val}')
            print(f'quant_val {float_val}')
            
        
#from modelopt.torch.export import export_hf_checkpoint

#with torch.inference_mode():
#    export_hf_checkpoint(
#        model,  # The quantized model.
#        export_dir="."
#    )
