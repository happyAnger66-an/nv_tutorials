from tensorrt_llm import LLM
from modeling_xiaoan import XiaoanModelForCausalLM

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

if __name__ == '__main__':
    main()