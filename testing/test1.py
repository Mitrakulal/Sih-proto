# test_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import re

# Load your trained model
print("Loading trained model...")
model_path = './log_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

print("Model loaded successfully!")
print("=" * 50)

def tokenize_request(raw_request):
    """Convert raw HTTP request to tokenized format"""
    # Replace numbers with <NUM>
    tokenized = re.sub(r'\d+', '<NUM>', raw_request)
    # Add spaces around special characters
    tokenized = re.sub(r'([^\s\w])', r' \1 ', tokenized)
    # Clean up extra spaces
    tokenized = ' '.join(tokenized.split())
    return tokenized

def generate_completion(prompt, max_length=50, temperature=0.7):
    """Generate text completion for a given prompt"""
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def calculate_perplexity(text):
    """Calculate perplexity (lower = more likely according to model)"""
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

# Test 1: Generate completions for partial requests
print("🔍 TEST 1: Generating Request Completions")
print("-" * 40)

test_prompts = [
    "GET /",
    "POST / login",
    "GET / products ? id =",
    "HTTP /",
    "GET / search ? q ="
]

for prompt in test_prompts:
    completion = generate_completion(prompt, max_length=30)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: {completion}")
    print()

print("=" * 50)

# Test 2: Compare perplexity between benign and malicious patterns
print("🔍 TEST 2: Perplexity Analysis (Anomaly Detection)")
print("-" * 40)

# Benign-like patterns (should have LOW perplexity)
benign_samples = [
    "GET / login HTTP / 1 . 1",
    "POST / products ? id = < NUM > HTTP / 1 . 1",
    "GET / search ? q = shoes HTTP / 1 . 1",
    "GET / cart HTTP / 1 . 1"
]

# Malicious-like patterns (should have HIGH perplexity)
malicious_samples = [
    "GET / login ? id = 1 ; DROP TABLE users HTTP / 1 . 1",
    "GET / products ? id = < script > alert ( 1 ) < / script > HTTP / 1 . 1",
    "GET / . . / . . / etc / passwd HTTP / 1 . 1",
    "GET / search ? q = ' OR ' 1 ' = ' 1 HTTP / 1 . 1"
]

print("📊 BENIGN REQUESTS (should have low perplexity):")
benign_perplexities = []
for sample in benign_samples:
    perp = calculate_perplexity(sample)
    benign_perplexities.append(perp)
    print(f"Perplexity: {perp:.2f} | Request: {sample}")

print("\n📊 MALICIOUS REQUESTS (should have high perplexity):")
malicious_perplexities = []
for sample in malicious_samples:
    perp = calculate_perplexity(sample)
    malicious_perplexities.append(perp)
    print(f"Perplexity: {perp:.2f} | Request: {sample}")

print("\n" + "=" * 50)

# Test 3: Summary statistics
print("🔍 TEST 3: Model Performance Summary")
print("-" * 40)

avg_benign = sum(benign_perplexities) / len(benign_perplexities)
avg_malicious = sum(malicious_perplexities) / len(malicious_perplexities)

print(f"Average Benign Perplexity: {avg_benign:.2f}")
print(f"Average Malicious Perplexity: {avg_malicious:.2f}")
print(f"Ratio (Malicious/Benign): {avg_malicious/avg_benign:.2f}")

if avg_malicious > avg_benign:
    print("✅ Good! Model assigns higher perplexity to malicious requests")
    print("   This suggests it can detect anomalies")
else:
    print("⚠️  Model needs more training - malicious requests have lower perplexity")

print("\n" + "=" * 50)

# Test 4: Interactive testing
print("🔍 TEST 4: Interactive Testing")
print("-" * 40)
print("Enter your own HTTP requests to test (or 'quit' to exit):")
print("Example format: GET / admin ? user = < NUM > HTTP / 1 . 1")

while True:
    user_input = input("\nEnter request: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if user_input:
        perp = calculate_perplexity(user_input)
        completion = generate_completion(user_input, max_length=40)
        
        print(f"Perplexity: {perp:.2f}")
        print(f"Completion: {completion}")
        
        if perp > avg_benign * 1.5:  # Threshold for anomaly
            print("🚨 ANOMALY DETECTED - High perplexity!")
        else:
            print("✅ Looks normal - Low perplexity")

print("Testing complete!")