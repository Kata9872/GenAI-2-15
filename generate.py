from transformers import pipeline

generator = pipeline(text-generation, model=gpt2)

prompt = input(enter prompt )

result = generator(
    prompt, min_length=30, max_length=50
)

print(answer, result[0][generated_text])