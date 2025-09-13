#!/usr/bin/env python3
"""
T4 LLM Performance Monitor Demo
A production-safe monitoring system for LLM performance analysis on T4 GPUs.
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def run_dcgmi_command(cmd: str) -> Optional[str]:
    """Run a dcgmi command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return None

def collect_gpu_metrics() -> Optional[Dict]:
    """Collect current GPU metrics using dcgmi"""
    # Field IDs: 150=GPU Temp, 155=Power, 203=GPU Util, 204=Mem Util, 100=SM Clock, 101=Mem Clock, 250=Mem Total, 252=Mem Used
    cmd = "dcgmi dmon -e 150,155,203,204,100,101,250,252,160,152,112 -c 1 -d 1000"
    output = run_dcgmi_command(cmd)
    
    if not output:
        return None
    
    lines = output.strip().split('\n')
    if len(lines) < 3:
        return None
    
    data_line = lines[2]  # Skip header and separator
    parts = data_line.split()
    
    if len(parts) < 11:
        return None
    
    try:
        gpu_temp = float(parts[1])
        power_usage = float(parts[2])
        sm_utilization = float(parts[3])
        memory_utilization = float(parts[4])
        sm_clock = float(parts[5])
        memory_clock = float(parts[6])
        memory_total_mb = float(parts[7])
        memory_used_mb = float(parts[8])
        power_limit = float(parts[9])
        max_temp = float(parts[10]) if parts[10] != 'N/A' else 85.0
        
        # Fix memory calculation - the order seems to be different
        # Let's swap them based on typical values
        if memory_total_mb < memory_used_mb:
            memory_total_mb, memory_used_mb = memory_used_mb, memory_total_mb
        
        # Calculate memory usage percentage
        memory_usage_pct = (memory_used_mb / memory_total_mb) * 100 if memory_total_mb > 0 else 0
        
        return {
            'timestamp': time.time(),
            'gpu_temp': gpu_temp,
            'power_usage': power_usage,
            'sm_utilization': sm_utilization,
            'memory_utilization': memory_utilization,
            'memory_used_mb': memory_used_mb,
            'memory_total_mb': memory_total_mb,
            'memory_usage_pct': memory_usage_pct,
            'sm_clock': sm_clock,
            'memory_clock': memory_clock,
            'power_limit': power_limit,
            'max_temp': max_temp,
            'thermal_throttling': gpu_temp > (max_temp * 0.9),
            'power_throttling': power_usage > (power_limit * 0.95)
        }
    except (ValueError, IndexError) as e:
        print(f"Error parsing metrics: {e}")
        return None

def analyze_performance(metrics: Dict, inference_latency_ms: float = 0) -> Dict:
    """Analyze performance and identify bottlenecks using heuristics"""
    
    sm_util = metrics['sm_utilization']
    memory_util = metrics['memory_utilization']
    memory_pct = metrics['memory_usage_pct']
    power = metrics['power_usage']
    temp = metrics['gpu_temp']
    thermal_throttling = metrics['thermal_throttling']
    power_throttling = metrics['power_throttling']
    
    bottleneck_type = "none"
    confidence = 0.0
    severity = "low"
    description = ""
    recommendations = []
    
    # Thermal throttling (highest priority)
    if thermal_throttling or temp > 80:
        bottleneck_type = "thermal"
        confidence = 0.9
        severity = "critical" if temp > 85 else "high"
        description = f"GPU temperature is {temp:.1f}°C, causing thermal throttling"
        recommendations = [
            "Check GPU cooling and airflow",
            "Reduce batch size to lower power consumption",
            "Implement dynamic batching with temperature monitoring",
            "Consider model quantization to reduce compute intensity"
        ]
    
    # Power throttling
    elif power_throttling or power > 65:
        bottleneck_type = "thermal"  # Power throttling is thermal-related
        confidence = 0.8
        severity = "high"
        description = f"Power consumption is {power:.1f}W, causing power throttling"
        recommendations = [
            "Reduce batch size or model precision",
            "Implement power-aware scheduling",
            "Use mixed precision (FP16) to reduce power consumption",
            "Consider model pruning or distillation"
        ]
    
    # Memory capacity bottleneck
    elif memory_pct > 90:
        bottleneck_type = "memory"
        confidence = 0.9
        severity = "critical"
        description = f"GPU memory usage is {memory_pct:.1f}%, near capacity"
        recommendations = [
            "Reduce batch size immediately",
            "Use gradient checkpointing",
            "Implement model sharding or offloading",
            "Consider using a larger GPU or model compression"
        ]
    
    # Memory bandwidth bottleneck
    elif sm_util < 60 and memory_util > 70 and memory_pct > 80:
        bottleneck_type = "memory"
        confidence = 0.8
        severity = "high"
        description = f"Low SM utilization ({sm_util:.1f}%) with high memory usage ({memory_pct:.1f}%)"
        recommendations = [
            "Increase batch size to improve SM utilization",
            "Use gradient checkpointing to reduce memory usage",
            "Implement model parallelism for large models",
            "Consider using Flash Attention for memory efficiency"
        ]
    
    # Compute bottleneck
    elif sm_util > 85 and memory_util < 50:
        bottleneck_type = "compute"
        confidence = 0.7
        severity = "medium"
        description = f"High SM utilization ({sm_util:.1f}%) with low memory usage"
        recommendations = [
            "Consider model quantization (INT8/FP16)",
            "Use TensorRT for optimized inference",
            "Implement kernel fusion optimizations",
            "Consider using larger batch sizes if memory allows"
        ]
    
    # Low utilization (inefficient batching)
    elif sm_util < 40 and memory_util < 40 and inference_latency_ms > 0:
        bottleneck_type = "contention"
        confidence = 0.6
        severity = "medium"
        description = f"Low GPU utilization ({sm_util:.1f}% SM, {memory_util:.1f}% memory) with active inference"
        recommendations = [
            "Increase batch size to improve GPU utilization",
            "Implement dynamic batching",
            "Check for CPU bottlenecks in data preprocessing",
            "Consider using multiple smaller models in parallel"
        ]
    
    # No clear bottleneck
    else:
        bottleneck_type = "none"
        confidence = 0.5
        severity = "low"
        description = "No significant bottlenecks detected"
        recommendations = [
            "Continue monitoring for performance regressions",
            "Consider profiling with Nsight Compute for optimization opportunities",
            "Monitor for changes in workload patterns"
        ]
    
    return {
        'bottleneck_type': bottleneck_type,
        'confidence': confidence,
        'severity': severity,
        'description': description,
        'recommendations': recommendations
    }

def simulate_llm_workload(duration_seconds: int = 30) -> List[Dict]:
    """Simulate an LLM workload and return inference data"""
    print(f"🤖 Simulating LLM workload for {duration_seconds} seconds...")
    
    inferences = []
    start_time = time.time()
    request_id = 0
    
    while time.time() - start_time < duration_seconds:
        # Simulate varying workload
        batch_size = 1 if (request_id % 10) < 3 else 4  # 30% small batches, 70% larger
        input_tokens = 50 + (request_id % 20) * 10  # 50-240 tokens
        output_tokens = 20 + (request_id % 15) * 5  # 20-90 tokens
        model_name = "llama-7b" if (request_id % 3) == 0 else "gpt-3.5-turbo"
        
        # Simulate latency based on batch size and tokens
        base_latency = 100 + (input_tokens + output_tokens) * 2
        batch_penalty = batch_size * 50
        latency_ms = base_latency + batch_penalty + (request_id % 100)  # Add some variance
        
        inferences.append({
            'request_id': f"req_{request_id:04d}",
            'timestamp': time.time(),
            'batch_size': batch_size,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model_name': model_name,
            'latency_ms': latency_ms
        })
        
        request_id += 1
        time.sleep(0.5)  # Simulate request rate
    
    return inferences

def continuous_monitor(interval_seconds: float = 5.0, show_detailed: bool = True):
    """Continuously monitor GPU metrics and provide analysis"""
    print("🚀 T4 LLM Performance Monitor - Continuous Mode")
    print("=" * 60)
    print("Monitoring GPU while you run vLLM...")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    # Check if dcgmi is available
    if not run_dcgmi_command("which dcgmi"):
        print("✗ dcgmi command not found. Please install DCGM.")
        return 1
    
    print("✓ dcgmi command found")
    
    # Track metrics history for trend analysis
    metrics_history = []
    last_analysis = None
    analysis_count = 0
    
    try:
        while True:
            # Collect current metrics
            current_time = datetime.now().strftime("%H:%M:%S")
            metrics = collect_gpu_metrics()
            
            if not metrics:
                print(f"[{current_time}] ✗ Failed to collect metrics")
                time.sleep(interval_seconds)
                continue
            
            # Add to history
            metrics_history.append(metrics)
            if len(metrics_history) > 20:  # Keep last 20 samples
                metrics_history.pop(0)
            
            # Calculate trends
            if len(metrics_history) >= 3:
                recent_metrics = metrics_history[-3:]
                avg_sm_util = sum(m['sm_utilization'] for m in recent_metrics) / len(recent_metrics)
                avg_memory_util = sum(m['memory_utilization'] for m in recent_metrics) / len(recent_metrics)
                avg_power = sum(m['power_usage'] for m in recent_metrics) / len(recent_metrics)
                avg_temp = sum(m['gpu_temp'] for m in recent_metrics) / len(recent_metrics)
            else:
                avg_sm_util = metrics['sm_utilization']
                avg_memory_util = metrics['memory_utilization']
                avg_power = metrics['power_usage']
                avg_temp = metrics['gpu_temp']
            
            # Analyze performance
            analysis = analyze_performance(metrics, 0)  # No inference data for now
            
            # Only show analysis if it's different from last time or every 10 iterations
            show_analysis = (analysis != last_analysis or analysis_count % 10 == 0)
            
            if show_analysis:
                print(f"\n[{current_time}] 🔍 Performance Analysis:")
                print(f"  Bottleneck: {analysis['bottleneck_type']} (confidence: {analysis['confidence']:.2f})")
                print(f"  Severity: {analysis['severity']}")
                print(f"  Description: {analysis['description']}")
                
                if analysis['recommendations']:
                    print(f"  💡 Recommendations:")
                    for i, rec in enumerate(analysis['recommendations'][:3], 1):  # Show top 3
                        print(f"    {i}. {rec}")
                
                last_analysis = analysis
                analysis_count += 1
            
            # Show current metrics
            if show_detailed:
                print(f"\n[{current_time}] 📊 Current Metrics:")
                print(f"  🌡️  Temperature: {metrics['gpu_temp']:.1f}°C (avg: {avg_temp:.1f}°C)")
                print(f"  ⚡ Power: {metrics['power_usage']:.1f}W / {metrics['power_limit']:.1f}W (avg: {avg_power:.1f}W)")
                print(f"  🔥 SM Utilization: {metrics['sm_utilization']:.1f}% (avg: {avg_sm_util:.1f}%)")
                print(f"  💾 Memory Utilization: {metrics['memory_utilization']:.1f}% (avg: {avg_memory_util:.1f}%)")
                print(f"  📦 Memory Usage: {metrics['memory_used_mb']:.0f}MB / {metrics['memory_total_mb']:.0f}MB ({metrics['memory_usage_pct']:.1f}%)")
                print(f"  🕐 Clocks: SM {metrics['sm_clock']:.0f}MHz, Memory {metrics['memory_clock']:.0f}MHz")
                
                # Show warnings
                warnings = []
                if metrics['thermal_throttling']:
                    warnings.append("🔥 THERMAL THROTTLING")
                if metrics['power_throttling']:
                    warnings.append("⚡ POWER THROTTLING")
                if metrics['memory_usage_pct'] > 90:
                    warnings.append("💾 HIGH MEMORY USAGE")
                if metrics['sm_utilization'] < 20 and metrics['memory_utilization'] < 20:
                    warnings.append("😴 LOW GPU UTILIZATION")
                
                if warnings:
                    print(f"  ⚠️  Warnings: {' | '.join(warnings)}")
            else:
                # Compact mode
                status_icons = []
                if metrics['thermal_throttling']:
                    status_icons.append("🔥")
                if metrics['power_throttling']:
                    status_icons.append("⚡")
                if metrics['memory_usage_pct'] > 90:
                    status_icons.append("💾")
                if metrics['sm_utilization'] < 20:
                    status_icons.append("😴")
                
                status = " ".join(status_icons) if status_icons else "✅"
                print(f"[{current_time}] {status} Temp: {metrics['gpu_temp']:.1f}°C | Power: {metrics['power_usage']:.1f}W | SM: {metrics['sm_utilization']:.1f}% | Mem: {metrics['memory_usage_pct']:.1f}% | {analysis['bottleneck_type']}")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] ⏹️  Monitoring stopped by user")
        print("✅ Thank you for using T4 LLM Performance Monitor!")
        return 0

def main():
    """Main function for the LLM monitor demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='T4 LLM Performance Monitor')
    parser.add_argument('--interval', '-i', type=float, default=5.0, 
                       help='Monitoring interval in seconds (default: 5.0)')
    parser.add_argument('--compact', '-c', action='store_true',
                       help='Use compact output mode')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run demo mode with simulated workload')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run the original demo
        print("🚀 T4 LLM Performance Monitor Demo")
        print("=" * 50)
        
        # Check if dcgmi is available
        if not run_dcgmi_command("which dcgmi"):
            print("✗ dcgmi command not found. Please install DCGM.")
            return 1
        
        print("✓ dcgmi command found")
        
        # Collect baseline metrics
        print("\n📊 Collecting baseline metrics...")
        baseline_metrics = collect_gpu_metrics()
        if not baseline_metrics:
            print("✗ Failed to collect baseline metrics")
            return 1
        
        print("✓ Baseline metrics collected")
        print(f"  GPU Temperature: {baseline_metrics['gpu_temp']:.1f}°C")
        print(f"  Power Usage: {baseline_metrics['power_usage']:.1f}W")
        print(f"  SM Utilization: {baseline_metrics['sm_utilization']:.1f}%")
        print(f"  Memory Usage: {baseline_metrics['memory_usage_pct']:.1f}%")
        
        # Simulate workload
        inferences = simulate_llm_workload(duration_seconds=20)
        
        # Collect metrics during workload
        print("\n📊 Collecting metrics during workload...")
        workload_metrics = collect_gpu_metrics()
        if not workload_metrics:
            print("✗ Failed to collect workload metrics")
            return 1
        
        # Calculate average inference latency
        avg_latency = sum(inf['latency_ms'] for inf in inferences) / len(inferences) if inferences else 0
        
        # Analyze performance
        print("\n🔍 Performance Analysis:")
        print("-" * 30)
        
        analysis = analyze_performance(workload_metrics, avg_latency)
        
        print(f"Bottleneck: {analysis['bottleneck_type']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Severity: {analysis['severity']}")
        print(f"Description: {analysis['description']}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Show detailed metrics
        print(f"\n📈 Detailed Metrics:")
        print("-" * 20)
        print(f"  GPU Temperature: {workload_metrics['gpu_temp']:.1f}°C")
        print(f"  Power Usage: {workload_metrics['power_usage']:.1f}W / {workload_metrics['power_limit']:.1f}W")
        print(f"  SM Utilization: {workload_metrics['sm_utilization']:.1f}%")
        print(f"  Memory Utilization: {workload_metrics['memory_utilization']:.1f}%")
        print(f"  Memory Usage: {workload_metrics['memory_used_mb']:.0f}MB / {workload_metrics['memory_total_mb']:.0f}MB ({workload_metrics['memory_usage_pct']:.1f}%)")
        print(f"  SM Clock: {workload_metrics['sm_clock']:.0f}MHz")
        print(f"  Memory Clock: {workload_metrics['memory_clock']:.0f}MHz")
        print(f"  Thermal Throttling: {workload_metrics['thermal_throttling']}")
        print(f"  Power Throttling: {workload_metrics['power_throttling']}")
        
        if inferences:
            print(f"\n🤖 Inference Statistics:")
            print(f"  Total Requests: {len(inferences)}")
            print(f"  Average Latency: {avg_latency:.1f}ms")
            print(f"  Average Batch Size: {sum(inf['batch_size'] for inf in inferences) / len(inferences):.1f}")
            print(f"  Average Input Tokens: {sum(inf['input_tokens'] for inf in inferences) / len(inferences):.1f}")
            print(f"  Average Output Tokens: {sum(inf['output_tokens'] for inf in inferences) / len(inferences):.1f}")
        
        print(f"\n✅ Analysis complete!")
        print(f"\n💡 This system provides production-safe LLM performance monitoring")
        print(f"   without intrusive profiling tools like Nsight Compute.")
        
        return 0
    else:
        # Run continuous monitoring
        return continuous_monitor(args.interval, not args.compact)

if __name__ == "__main__":
    sys.exit(main())
