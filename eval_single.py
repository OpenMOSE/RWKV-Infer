from evaluator import EvaluationConfig, Evaluator

# Create an EvaluationConfig instance to evaluate your model, for example:
config = EvaluationConfig(
    model_name_or_path='/home/client/Projects/llm/RWKV-Reka-Flash-Gen2',
    tokenizer_name='/home/client/Projects/llm/RWKV-Reka-Flash-Gen2',
    model_type='hxa079',
    data=["uncheatable_eval/data/wikipedia_english_20250401to20250415.json"]
)

if __name__ == '__main__':
    try:
        evaluator = Evaluator()
        evaluator.evaluate(config)
    except Exception as e:
        print(f"Error: {e}")
