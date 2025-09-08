#!/usr/bin/env python3
"""
Random Requests Generator for vLLM Server
Generates random completion requests to test the vLLM server and create metrics for Grafana.
"""

import requests
import json
import random
import time
import argparse
from typing import List, Dict, Any

# Sample prompts for different types of requests
PROMPTS = [
    "Ashgabat is a",
    "The capital of Turkmenistan is",
    "Machine learning is",
    "Artificial intelligence can",
    "Python programming is",
    "Docker containers are",
    "Cloud computing provides",
    "Data science involves",
    "Neural networks are",
    "Deep learning models",
    "Natural language processing",
    "Computer vision applications",
    "The future of AI is",
    "Blockchain technology",
    "Quantum computing will",
    "Cybersecurity measures",
    "Software development best practices",
    "Database optimization techniques",
    "Web application architecture",
    "Mobile app development"
]

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
]

def generate_random_request() -> Dict[str, Any]:
    """Generate a random completion request."""
    return {
        "model": random.choice(MODELS),
        "prompt": random.choice(PROMPTS),
        "max_tokens": random.randint(50, 1000),
        "temperature": round(random.uniform(0.1, 1.5), 1),
        "top_p": round(random.uniform(0.8, 1.0), 2),
        "frequency_penalty": round(random.uniform(0.0, 0.5), 1),
        "presence_penalty": round(random.uniform(0.0, 0.5), 1)
    }

def send_request(url: str, request_data: Dict[str, Any]) -> bool:
    """Send a request to the vLLM server."""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful - Model: {request_data['model']}, "
                  f"Prompt: '{request_data['prompt'][:30]}...', "
                  f"Tokens: {request_data['max_tokens']}")
            return True
        else:
            print(f"❌ Request failed - Status: {response.status_code}, "
                  f"Response: {response.text[:100]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate random requests to vLLM server")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions",
                       help="vLLM server URL (default: http://localhost:8000/v1/completions)")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of requests to send (default: 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Interval between requests in seconds (default: 1.0)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuously until interrupted")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting random requests generator")
    print(f"📡 Target URL: {args.url}")
    print(f"📊 Request count: {'∞ (continuous)' if args.continuous else args.count}")
    print(f"⏱️  Interval: {args.interval}s")
    print("-" * 60)
    
    successful_requests = 0
    failed_requests = 0
    
    try:
        if args.continuous:
            print("🔄 Running in continuous mode. Press Ctrl+C to stop.")
            request_count = 0
            while True:
                request_count += 1
                request_data = generate_random_request()
                print(f"\n📝 Request #{request_count}:")
                
                if send_request(args.url, request_data):
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                time.sleep(args.interval)
        else:
            for i in range(args.count):
                request_data = generate_random_request()
                print(f"\n📝 Request #{i+1}/{args.count}:")
                
                if send_request(args.url, request_data):
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                if i < args.count - 1:  # Don't sleep after the last request
                    time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY:")
    print(f"✅ Successful requests: {successful_requests}")
    print(f"❌ Failed requests: {failed_requests}")
    print(f"📊 Success rate: {(successful_requests/(successful_requests+failed_requests)*100):.1f}%" if (successful_requests+failed_requests) > 0 else "N/A")
    print("=" * 60)

if __name__ == "__main__":
    main()
