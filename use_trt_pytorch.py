import torch
import torchvision
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

# 1. 导出PyTorch模型到ONNX
def export_to_onnx():
    # 加载PyTorch模型
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        "resnet50.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully!")

def build_engine(onnx_file_path, engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX model.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 构建配置
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16加速
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    
    return engine

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 分配输入输出内存
        self.inputs, self.outputs, self.bindings = [], [], []

        i = 0 
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            print(f'binding: {binding}') 
            # 分配设备内存
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))
            #if self.engine.binding_is_input(binding):
            if i == 0:
                self.inputs.append({'device': device_mem, 'shape': self.engine.get_tensor_shape(binding), 'dtype': dtype})
            else:
                self.outputs.append({'device': device_mem, 'shape': self.engine.get_tensor_shape(binding), 'dtype': dtype})
            i+=1
    
    def infer(self, input_data):
        # 传输输入数据到GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], input_data, self.stream)
        
        # 执行推理
        self.context.execute_v2(bindings=self.bindings)
        
        # 传输输出数据到CPU
        output_data = np.zeros(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh_async(output_data, self.outputs[0]['device'], self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        return output_data

if __name__ == "__main__":
    import sys

    # 构建trt engine
    trt_inference = TensorRTInference(sys.argv[1])
    
    # 准备输入数据
    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    
    # 执行推理
    output = trt_inference.infer(input_data)
    print(f"Output shape: {output.shape} ")

#    export_to_onnx()
#    build_engine(sys.argv[1], sys.argv[2])