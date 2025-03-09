from vllm import LLM, SamplingParams
if __name__ == '__main__':
 model_path = "DeepSeek-R1-Distill-Qwen-7B"
 model = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True, max_model_len=10000, enforce_eager=True, 
 gpu_memory_utilization=0.5, block_size=32)
 sampling_params = SamplingParams(temperature=0.2, max_tokens=1, prompt_logprobs=20)
 
 prompt = "今天天气怎么样?"
 response = model.generate(prompt, sampling_params, use_tqdm=False)[0]
 print(response, '\n\n', response.outputs)