#!/usr/bin/env python3
"""
T4 LLM Performance Monitor
A production-safe monitoring system that combines DCGM metrics with intelligent heuristics
to diagnose LLM performance bottlenecks without intrusive profiling.
"""

import sys
import os
import time
import json
import subprocess
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import statistics

# Add DCGM Python bindings to the path
sys.path.insert(0, '/usr/share/datacenter-gpu-manager-4/bindings/python3')

try:
    import pydcgm
    import dcgm_structs
    import dcgm_fields
    DCGM_AVAILABLE = True
except ImportError:
    DCGM_AVAILABLE = False
    print("Warning: DCGM Python bindings not available. Using dcgmi command-line tool.")

@dataclass
class GPUMetrics:
    """GPU metrics snapshot"""
    timestamp: float
    gpu_id: int
    gpu_temp: float
    power_usage: float
    sm_utilization: float
    tensor_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    sm_clock: float
    memory_clock: float
    sm_max_clock: float
    memory_max_clock: float
    power_limit: float
    thermal_throttling: bool = False
    power_throttling: bool = False
    ecc_errors: int = 0

@dataclass
class InferenceRequest:
    """LLM inference request tracking"""
    request_id: str
    timestamp: float
    batch_size: int
    input_tokens: int
    output_tokens: int
    model_name: str
    latency_ms: float
    gpu_id: int

@dataclass
class PerformanceAnalysis:
    """Performance analysis result"""
    timestamp: float
    gpu_id: int
    bottleneck_type: str  # 'compute', 'memory', 'thermal', 'contention', 'none'
    confidence: float  # 0.0 to 1.0
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    recommendations: List[str]
    metrics_summary: Dict

class DCGMMetricsCollector:
    """Collects GPU metrics using DCGM"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.dcgm_handle = None
        self.group = None
        self.field_group = None
        self._init_dcgm()
    
    def _init_dcgm(self):
        """Initialize DCGM connection"""
        if not DCGM_AVAILABLE:
            return
        
        try:
            self.dcgm_handle = pydcgm.DcgmHandle()
            system = self.dcgm_handle.GetSystem()
            discovery = system.discovery
            
            # Create monitoring group
            self.group = pydcgm.DcgmGroup(self.dcgm_handle, f"llm_monitor_{self.gpu_id}")
            self.group.AddGpu(self.gpu_id)
            
            # Define fields to monitor
            fields_to_monitor = [
                dcgm_fields.DCGM_FI_DEV_GPU_TEMP,           # 150
                dcgm_fields.DCGM_FI_DEV_POWER_USAGE,        # 155
                dcgm_fields.DCGM_FI_DEV_GPU_UTIL,           # 203
                dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,      # 204
                dcgm_fields.DCGM_FI_DEV_SM_CLOCK,           # 100
                dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,          # 101
                dcgm_fields.DCGM_FI_DEV_FB_USED,            # 252
                dcgm_fields.DCGM_FI_DEV_FB_TOTAL,           # 250
                dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT,   # 160
                dcgm_fields.DCGM_FI_DEV_GPU_MAX_OP_TEMP,    # 152
                dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS, # 112
            ]
            
            self.field_group = pydcgm.DcgmFieldGroup(
                self.dcgm_handle, "llm_fields", fields_to_monitor
            )
            
            # Start monitoring
            self.group.WatchFields(self.field_group, 1000000, 10.0, 0)
            print(f"✓ DCGM monitoring initialized for GPU {self.gpu_id}")
            
        except Exception as e:
            print(f"✗ Failed to initialize DCGM: {e}")
            self.dcgm_handle = None
    
    def collect_metrics(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics"""
        if self.dcgm_handle and self.group and self.field_group:
            return self._collect_dcgm_metrics()
        else:
            return self._collect_dcgmi_metrics()
    
    def _collect_dcgm_metrics(self) -> Optional[GPUMetrics]:
        """Collect metrics using DCGM Python bindings"""
        try:
            values = self.group.GetLatest(self.field_group)
            
            metrics = {
                'timestamp': time.time(),
                'gpu_id': self.gpu_id,
                'gpu_temp': 0.0,
                'power_usage': 0.0,
                'sm_utilization': 0.0,
                'tensor_utilization': 0.0,
                'memory_utilization': 0.0,
                'memory_used_mb': 0.0,
                'memory_total_mb': 0.0,
                'sm_clock': 0.0,
                'memory_clock': 0.0,
                'sm_max_clock': 0.0,
                'memory_max_clock': 0.0,
                'power_limit': 0.0,
                'thermal_throttling': False,
                'power_throttling': False,
                'ecc_errors': 0
            }
            
            for value in values:
                field_id = value.fieldId
                field_value = value.value
                
                if field_id == dcgm_fields.DCGM_FI_DEV_GPU_TEMP:
                    metrics['gpu_temp'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_POWER_USAGE:
                    metrics['power_usage'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_GPU_UTIL:
                    metrics['sm_utilization'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL:
                    metrics['memory_utilization'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_SM_CLOCK:
                    metrics['sm_clock'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_MEM_CLOCK:
                    metrics['memory_clock'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_FB_USED:
                    metrics['memory_used_mb'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_FB_TOTAL:
                    metrics['memory_total_mb'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT:
                    metrics['power_limit'] = float(field_value)
                elif field_id == dcgm_fields.DCGM_FI_DEV_GPU_MAX_OP_TEMP:
                    max_temp = float(field_value)
                    metrics['thermal_throttling'] = metrics['gpu_temp'] > (max_temp * 0.9)
                elif field_id == dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS:
                    clock_reasons = int(field_value)
                    # Check for thermal and power throttling reasons
                    thermal_reason = 0x20  # DCGM_CLOCKS_EVENT_REASON_SW_THERMAL
                    power_reason = 0x4     # DCGM_CLOCKS_EVENT_REASON_SW_POWER_CAP
                    metrics['thermal_throttling'] = bool(clock_reasons & thermal_reason)
                    metrics['power_throttling'] = bool(clock_reasons & power_reason)
            
            return GPUMetrics(**metrics)
            
        except Exception as e:
            print(f"Error collecting DCGM metrics: {e}")
            return None
    
    def _collect_dcgmi_metrics(self) -> Optional[GPUMetrics]:
        """Collect metrics using dcgmi command-line tool"""
        try:
            # Get basic metrics
            cmd = f"dcgmi dmon -e 150,155,203,204,100,101,250,252,160,152,112 -c 1 -d 1000"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 3:
                return None
            
            data_line = lines[2]  # Skip header and separator
            parts = data_line.split()
            
            if len(parts) < 11:
                return None
            
            # Parse the data
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
            
            # Get clock reasons for throttling detection
            clock_reasons = 0
            try:
                cmd2 = f"dcgmi dmon -e 112 -c 1 -d 1000"
                result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=5)
                if result2.returncode == 0:
                    lines2 = result2.stdout.strip().split('\n')
                    if len(lines2) >= 3:
                        clock_reasons = int(lines2[2].split()[1])
            except:
                pass
            
            thermal_reason = 0x20
            power_reason = 0x4
            thermal_throttling = bool(clock_reasons & thermal_reason)
            power_throttling = bool(clock_reasons & power_reason)
            
            return GPUMetrics(
                timestamp=time.time(),
                gpu_id=self.gpu_id,
                gpu_temp=gpu_temp,
                power_usage=power_usage,
                sm_utilization=sm_utilization,
                tensor_utilization=0.0,  # Not available via dcgmi
                memory_utilization=memory_utilization,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                sm_clock=sm_clock,
                memory_clock=memory_clock,
                sm_max_clock=1590.0,  # T4 typical max
                memory_max_clock=5001.0,  # T4 typical max
                power_limit=power_limit,
                thermal_throttling=thermal_throttling,
                power_throttling=power_throttling,
                ecc_errors=0
            )
            
        except Exception as e:
            print(f"Error collecting dcgmi metrics: {e}")
            return None

class LLMPerformanceAnalyzer:
    """Analyzes LLM performance using heuristics and metrics correlation"""
    
    def __init__(self, history_window: int = 60):
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.inference_history = deque(maxlen=history_window * 10)  # More inference data
        self.analysis_history = deque(maxlen=history_window)
    
    def add_metrics(self, metrics: GPUMetrics):
        """Add new GPU metrics"""
        self.metrics_history.append(metrics)
    
    def add_inference(self, inference: InferenceRequest):
        """Add new inference request"""
        self.inference_history.append(inference)
    
    def analyze_performance(self, gpu_id: int = 0) -> Optional[PerformanceAnalysis]:
        """Analyze current performance and identify bottlenecks"""
        if len(self.metrics_history) < 5:
            return None
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        recent_inferences = [inf for inf in self.inference_history 
                           if inf.timestamp > time.time() - 60 and inf.gpu_id == gpu_id]
        
        # Calculate averages
        avg_sm_util = statistics.mean([m.sm_utilization for m in recent_metrics])
        avg_memory_util = statistics.mean([m.memory_utilization for m in recent_metrics])
        avg_power = statistics.mean([m.power_usage for m in recent_metrics])
        avg_temp = statistics.mean([m.gpu_temp for m in recent_metrics])
        avg_memory_used = statistics.mean([m.memory_used_mb for m in recent_metrics])
        memory_usage_pct = (avg_memory_used / recent_metrics[0].memory_total_mb) * 100
        
        # Check for throttling
        thermal_throttling = any(m.thermal_throttling for m in recent_metrics)
        power_throttling = any(m.power_throttling for m in recent_metrics)
        
        # Calculate inference latency statistics
        if recent_inferences:
            avg_latency = statistics.mean([inf.latency_ms for inf in recent_inferences])
            latency_std = statistics.stdev([inf.latency_ms for inf in recent_inferences]) if len(recent_inferences) > 1 else 0
        else:
            avg_latency = 0
            latency_std = 0
        
        # Heuristic analysis
        bottleneck_type, confidence, severity, description, recommendations = self._analyze_bottlenecks(
            avg_sm_util, avg_memory_util, avg_power, avg_temp, memory_usage_pct,
            thermal_throttling, power_throttling, avg_latency, latency_std, recent_inferences
        )
        
        metrics_summary = {
            'avg_sm_utilization': avg_sm_util,
            'avg_memory_utilization': avg_memory_util,
            'avg_power_usage': avg_power,
            'avg_temperature': avg_temp,
            'memory_usage_percent': memory_usage_pct,
            'thermal_throttling': thermal_throttling,
            'power_throttling': power_throttling,
            'avg_inference_latency_ms': avg_latency,
            'latency_std_ms': latency_std,
            'inference_count': len(recent_inferences)
        }
        
        analysis = PerformanceAnalysis(
            timestamp=time.time(),
            gpu_id=gpu_id,
            bottleneck_type=bottleneck_type,
            confidence=confidence,
            severity=severity,
            description=description,
            recommendations=recommendations,
            metrics_summary=metrics_summary
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _analyze_bottlenecks(self, sm_util, memory_util, power, temp, memory_pct, 
                           thermal_throttling, power_throttling, avg_latency, latency_std, 
                           recent_inferences) -> Tuple[str, float, str, str, List[str]]:
        """Analyze bottlenecks using heuristics"""
        
        recommendations = []
        confidence = 0.0
        severity = "low"
        description = ""
        
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
        
        # Low utilization (inefficient batching)
        elif sm_util < 40 and memory_util < 40 and recent_inferences:
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
        
        # High latency variance
        elif latency_std > avg_latency * 0.5 and recent_inferences:
            bottleneck_type = "contention"
            confidence = 0.7
            severity = "medium"
            description = f"High inference latency variance (std: {latency_std:.1f}ms, avg: {avg_latency:.1f}ms)"
            recommendations = [
                "Implement request queuing and batching",
                "Check for resource contention with other processes",
                "Use consistent batch sizes",
                "Monitor CPU and I/O bottlenecks"
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
        
        return bottleneck_type, confidence, severity, description, recommendations

class LLMMonitor:
    """Main LLM performance monitoring system"""
    
    def __init__(self, gpu_id: int = 0, collection_interval: float = 1.0):
        self.gpu_id = gpu_id
        self.collection_interval = collection_interval
        self.metrics_collector = DCGMMetricsCollector(gpu_id)
        self.analyzer = LLMPerformanceAnalyzer()
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"✓ Started LLM performance monitoring for GPU {self.gpu_id}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("✓ Stopped LLM performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_metrics()
                if metrics:
                    self.analyzer.add_metrics(metrics)
                
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def add_inference_request(self, request_id: str, batch_size: int, 
                            input_tokens: int, output_tokens: int, 
                            model_name: str, latency_ms: float):
        """Add an inference request for correlation analysis"""
        inference = InferenceRequest(
            request_id=request_id,
            timestamp=time.time(),
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            latency_ms=latency_ms,
            gpu_id=self.gpu_id
        )
        self.analyzer.add_inference(inference)
    
    def get_current_analysis(self) -> Optional[PerformanceAnalysis]:
        """Get current performance analysis"""
        return self.analyzer.analyze_performance(self.gpu_id)
    
    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary"""
        if not self.analyzer.metrics_history:
            return {}
        
        latest = self.analyzer.metrics_history[-1]
        return {
            'timestamp': latest.timestamp,
            'gpu_id': latest.gpu_id,
            'gpu_temp': latest.gpu_temp,
            'power_usage': latest.power_usage,
            'sm_utilization': latest.sm_utilization,
            'memory_utilization': latest.memory_utilization,
            'memory_usage_percent': (latest.memory_used_mb / latest.memory_total_mb) * 100,
            'thermal_throttling': latest.thermal_throttling,
            'power_throttling': latest.power_throttling
        }

def simulate_llm_workload(monitor: LLMMonitor, duration_seconds: int = 60):
    """Simulate an LLM workload for testing"""
    print(f"🤖 Simulating LLM workload for {duration_seconds} seconds...")
    
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
        
        # Add the inference request
        monitor.add_inference_request(
            request_id=f"req_{request_id:04d}",
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            latency_ms=latency_ms
        )
        
        request_id += 1
        time.sleep(0.5)  # Simulate request rate

def main():
    """Main function for testing the LLM monitor"""
    print("🚀 T4 LLM Performance Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = LLMMonitor(gpu_id=0, collection_interval=2.0)
    monitor.start_monitoring()
    
    try:
        # Let it collect some baseline metrics
        print("📊 Collecting baseline metrics...")
        time.sleep(10)
        
        # Simulate workload
        simulate_llm_workload(monitor, duration_seconds=30)
        
        # Get analysis
        print("\n🔍 Performance Analysis:")
        print("-" * 30)
        
        analysis = monitor.get_current_analysis()
        if analysis:
            print(f"Bottleneck: {analysis.bottleneck_type}")
            print(f"Confidence: {analysis.confidence:.2f}")
            print(f"Severity: {analysis.severity}")
            print(f"Description: {analysis.description}")
            print("\nRecommendations:")
            for i, rec in enumerate(analysis.recommendations, 1):
                print(f"  {i}. {rec}")
            
            print(f"\nMetrics Summary:")
            for key, value in analysis.metrics_summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("No analysis available yet")
        
        # Show current metrics
        print(f"\n📈 Current Metrics:")
        print("-" * 20)
        metrics = monitor.get_metrics_summary()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitor...")
    finally:
        monitor.stop_monitoring()
        print("✅ Monitor stopped")

if __name__ == "__main__":
    main()
