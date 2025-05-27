from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model_and_tokenizer():
    """
    Initialize the RWKV-7 model and tokenizer.
    Returns: tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        'fla-hub/rwkv7-1.5B-world',
        trust_remote_code=True
    )
    model = model.to('cuda')
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'fla-hub/rwkv7-1.5B-world',
        trust_remote_code=True
    )
    
    return model, tokenizer

def count_tokens(text: str, tokenizer) -> tuple:
    """
    Count the number of tokens in the input text.
    
    Args:
        text (str): Input text to tokenize
        tokenizer: The RWKV tokenizer instance
    
    Returns:
        tuple: (token_count, tokens, token_ids)
    """
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    # Get token IDs
    token_ids = tokenizer.encode(text)
    # Get the count
    token_count = len(tokens)
    
    return token_count, tokens, token_ids

# Example usage
if __name__ == "__main__":
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    
    # Example texts
    example_texts = [
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    ]
    # Garbage is 24 tokens (26 with backticks), task_description is 30 tokens, Needle is 17-19 tokens, question is 10 tokens. 
    # Garbage is 89 char (95 with backticks), task_description is 148 char, needle is 55-58 char, question is 37 char 
    # Process each example
    for text in example_texts:
        count, tokens, ids = count_tokens(text, tokenizer)
        print(f"\nInput text: {text}")
        print(f"Token count: {count}")
        # print(f"Tokens: {tokens}")
        