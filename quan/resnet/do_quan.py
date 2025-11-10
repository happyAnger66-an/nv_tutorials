import torch
from torch.utils.data import DataLoader
from torch import nn    

import modelopt.torch.quantization as mtq
from modelopt.torch.export import (
    export_hf_checkpoint,
    export_tensorrt_llm_checkpoint,
    get_model_type,
)

from data_ref import training_data 
from model_def import NeuralNetwork, test
from device_sel import device

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
# Setup the model
#model = get_model()

# Select quantization config
#config = mtq.INT8_SMOOTHQUANT_CFG
config = mtq.INT8_DEFAULT_CFG

# Quantization need calibration data. Setup calibration data loader
# An example of creating a calibration data loader looks like the following:
#data_loader = get_dataloader(num_samples=calib_size)

batch_size = 64
data_loader = DataLoader(training_data, batch_size=batch_size)

from hooks.torch_hooks import hook_module_outputs
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
loss_fn = nn.CrossEntropyLoss()
test(data_loader, model, loss_fn)

import copy
q_model = copy.deepcopy(model)
torch.onnx.export(
    model,
    torch.randn((64, 1, 28, 28), dtype=torch.float32).cuda(),
    "./no_action_expert.onnx",
    export_params=True,
    dynamo=True,
    do_constant_folding=True,
)
    

# Quantize the model and perform calibration (PTQ)
q_model = mtq.quantize(q_model, config, forward_loop)
import modelopt.torch.opt as mto
#mto.save(q_model, 'quan_model.pth', dtype=torch.int8)

print(f'no quantize model: {model}')
print(f'quantize model: {q_model}')

current_round = 1
print(f'after quantize\n')
test(data_loader, q_model, loss_fn)
mtq.print_quant_summary(q_model)
torch.onnx.export(
    q_model,
    (torch.randn((64, 1, 28, 28), dtype=torch.float32).cuda()),
    "./action_expert.onnx",
    export_params=True,
#    dynamo=True,
    input_names=["x"],
    output_names=["output"],
    do_constant_folding=True,
    )
    
before_hooks_funcs = hook_module_outputs(model, float_outputs)
after_hooks_funcs = hook_module_outputs(q_model, quant_outputs)
mse_loss_fn = nn.MSELoss()

import wandb
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="happyanger66-nio",
    # Set the wandb project where this run will be logged.
    project="my-quantization-resnet",
    # Track hyperparameters and run metadata.
    config={
        "architecture": "resnet",
        "method": "INT8_SMOOTHQUANT",
    },
)

for i, batch in enumerate(data_loader):
    x, y = batch
    X = x.to(device)
   
    float_outputs.clear() 
    quant_outputs.clear()
    
    model(X)
    q_model(X)
   
    for layer_name in float_outputs:
        if layer_name in quant_outputs:
            float_val = float_outputs[layer_name].cpu()
            quant_val = quant_outputs[layer_name].cpu()
            mse = mse_loss_fn(float_val, quant_val)
            run.log({f"{layer_name}_mse": mse})

run.finish()        
#from modelopt.torch.export import export_hf_checkpoint

#with torch.inference_mode():
#    export_hf_checkpoint(
#        model,  # The quantized model.
#        export_dir="."
#    )
