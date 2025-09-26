import random

methods = ["GET", "POST"]
benign_paths = [
    "/", 
    "/login", 
    "/products?id=123", 
    "/cart", 
    "/profile?user=45", 
    "/search?q=shoes"
]

malicious_paths = [
    "/login?id=1;DROP TABLE users",        # SQL Injection
    "/products?id=<script>alert(1)</script>",  # XSS
    "/../../etc/passwd",                   # Path Traversal
    "/search?q=' OR '1'='1",               # SQL Injection
    "/admin?user=123&token=abcdefg<script>" # XSS
]

def generate_log(ip, method, path, status="200", size="1234", time="22/Sep/2025:12:10:15 +0530"):
    return f'{ip} - - [{time}] "{method} {path} HTTP/1.1" {status} {size}\n'

# Generate benign logs
with open("benign_logs.txt", "w") as f:
    for i in range(50):
        ip = "127.0.0.1"
        method = random.choice(methods)
        path = random.choice(benign_paths)
        f.write(generate_log(ip, method, path))

# Generate malicious logs
with open("malicious_logs.txt", "w") as f:
    for i in range(20):
        ip = "127.0.0.1"
        method = random.choice(methods)
        path = random.choice(malicious_paths)
        f.write(generate_log(ip, method, path))
