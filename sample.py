import torch
import argparse
from config import GPTConfig
from gpt import GPTLanguageModel

def load_model_and_tokenizer(config_path=None):
    config = GPTConfig()
    if config_path:
        pass
    
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # tokenize 
    chars = sorted(list(set(text)))
    config.vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # init model
    model = GPTLanguageModel(config)
    model.to(config.device)
    
    try:
        model.load(config.model_save_path)
        print(f"Model loaded from {config.model_save_path}")
    except:
        print("No saved model found. Using untrained model.")
    
    return model, encode, decode, config

def generate_text(model, decode, prompt="", max_tokens=1000, config=None):
    if not prompt:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    else:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=config.device)
    
    output = model.generate(context, max_new_tokens=max_tokens)
    generated_text = decode(output[0].tolist())
    
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with TopGPT")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of tokens to generate")
    parser.add_argument("--output", type=str, default=None, help="Output file path (prints to console if not specified)")
    args = parser.parse_args()
    
    model, encode, decode, config = load_model_and_tokenizer()
    
    generated_text = generate_text(model, decode, args.prompt, args.max_tokens, config)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(generated_text)
        print(f"Generated text saved to {args.output}")
    else:
        print(generated_text)