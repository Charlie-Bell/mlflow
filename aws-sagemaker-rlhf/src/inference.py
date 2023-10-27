import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from instruction_pipeline import InstructionTextGenerationPipeline

def model_fn(model_dir):
    print(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
    print(model)

    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    model.eval()

    # Instruction pipeline
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    # return dictonary, which will be json serializable
    return generate_text(data['inputs'])