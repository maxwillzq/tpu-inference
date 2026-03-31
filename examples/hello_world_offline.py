import os
from vllm import LLM, SamplingParams

def main():
    # Use a small verified model from support matrix
    model_id = "Qwen/Qwen3-4B"
    
    export_path = os.environ.get("GOOGLE_EXPORT_MODEL_PATH")
    if export_path:
        print(f"Export mode enabled. Target path: {export_path}")
    
    # Use load_format="dummy" to skip downloading real weights if only exporting
    load_format = "dummy" if export_path else "auto"
    
    print(f"Loading/Initializing model: {model_id} (load_format={load_format})...")
    
    try:
        # Initializing LLM will trigger capture_model and our export logic if GOOGLE_EXPORT_MODEL_PATH is set
        llm = LLM(model=model_id, max_model_len=512, load_format=load_format)
    except RuntimeError as e:
        if "Export complete" in str(e):
            print(f"\n[SUCCESS] Model backbone exported to {export_path}")
            return
        else:
            raise e

    if export_path:
        print(f"\n[SUCCESS] Model backbone exported to {export_path}")
        return

    # Standard Inference
    prompt = "Hello, my name is"
    sampling_params = SamplingParams(max_tokens=20)
    
    print(f"\nRunning inference for prompt: '{prompt}'...")
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
