import random
import re

# Load logs
with open("C:\\Users\\kulal\\Desktop\\try proto\\dummy\\benign_logs.txt", "r") as f:
    logs = f.readlines()


# Pick 10 random logs
random_logs = random.sample(logs, min(50, len(logs)))

# Clear/create tokens.txt
with open("tokens.txt", "w") as f:
    pass

for i, raw_log in enumerate(random_logs, 1):
    raw_log = raw_log.strip()
    # print(f"\n--- Log {i} ---")
    # print("Raw Log:", raw_log)

    # Robust Parsing: find method, path, protocol
    match = re.search(r'"(\w+)\s+([^"]+)\s+(HTTP/\d\.\d)"', raw_log)
    if match:
        method, path, protocol = match.groups()
        parsed = {"method": method, "path": path, "protocol": protocol}
        # print("Parsed:", parsed)

        # Normalization: replace numbers with <NUM>
        normalized_path = re.sub(r'\d+', '<NUM>', path)
        normalized_request = f"{method} {normalized_path} {protocol}"
        #  print("Normalized:", normalized_request)

        # Tokenization: split into tokens
        tokens = re.findall(r'\w+|[^\s\w]', normalized_request)
        # print("Tokenized:", tokens)

        # Save to tokens.txt
        with open("tokens.txt", "a") as f:
            f.write(" ".join(tokens) + "\n")
    # else:
    #     print("⚠️ Regex did not match this log!")
