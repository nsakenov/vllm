# T4 LLM Performance Monitor - Usage Guide

## 🚀 Quick Start

### Continuous Monitoring (Recommended for vLLM)
```bash
# Start monitoring with 5-second intervals (default)
python3 llm_monitor_demo.py

# Use compact mode for less verbose output
python3 llm_monitor_demo.py --compact

# Custom monitoring interval (e.g., 2 seconds)
python3 llm_monitor_demo.py --interval 2.0

# Compact mode with 3-second intervals
python3 llm_monitor_demo.py --compact --interval 3.0
```

### Demo Mode (Testing)
```bash
# Run demo with simulated workload
python3 llm_monitor_demo.py --demo
```

## 📊 Output Modes

### Detailed Mode (Default)
Shows comprehensive metrics and analysis:
```
[21:40:34] 🔍 Performance Analysis:
  Bottleneck: memory (confidence: 0.90)
  Severity: critical
  Description: GPU memory usage is 95.2%, near capacity
  💡 Recommendations:
    1. Reduce batch size immediately
    2. Use gradient checkpointing
    3. Implement model sharding or offloading

[21:40:34] 📊 Current Metrics:
  🌡️  Temperature: 42.0°C (avg: 41.5°C)
  ⚡ Power: 65.0W / 70.0W (avg: 64.2W)
  🔥 SM Utilization: 45.0% (avg: 43.2%)
  💾 Memory Utilization: 85.0% (avg: 82.1%)
  📦 Memory Usage: 15200MB / 16384MB (92.8%)
  🕐 Clocks: SM 1200MHz, Memory 5000MHz
  ⚠️  Warnings: 💾 HIGH MEMORY USAGE
```

### Compact Mode
Shows essential metrics in one line:
```
[21:40:34] ✅ Temp: 42.0°C | Power: 65.0W | SM: 45.0% | Mem: 92.8% | memory
[21:40:39] 🔥💾 Temp: 85.0°C | Power: 70.0W | SM: 20.0% | Mem: 95.0% | thermal
[21:40:44] ⚡ Temp: 75.0°C | Power: 70.0W | SM: 60.0% | Mem: 80.0% | thermal
```

## 🔍 Understanding the Output

### Status Icons
- `✅` - All good, no issues detected
- `🔥` - Thermal throttling detected
- `⚡` - Power throttling detected  
- `💾` - High memory usage (>90%)
- `😴` - Low GPU utilization (<20%)

### Bottleneck Types
- **`none`** - No significant bottlenecks
- **`thermal`** - Temperature or power throttling
- **`memory`** - Memory capacity or bandwidth issues
- **`compute`** - GPU compute bound
- **`contention`** - Low utilization, inefficient batching

### Severity Levels
- **`low`** - Minor optimization opportunities
- **`medium`** - Noticeable performance impact
- **`high`** - Significant performance degradation
- **`critical`** - Immediate action required

## 🎯 Using with vLLM

### 1. Start the Monitor
```bash
# In one terminal, start monitoring
python3 llm_monitor_demo.py --compact --interval 3.0
```

### 2. Run vLLM
```bash
# In another terminal, run your vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model your-model-name \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8
```

### 3. Monitor Performance
The monitor will show:
- Real-time GPU metrics
- Bottleneck detection
- Actionable recommendations
- Performance warnings

## 📈 Interpreting Results

### High Memory Usage
```
💾 HIGH MEMORY USAGE
Bottleneck: memory
Recommendations:
  1. Reduce batch size immediately
  2. Use gradient checkpointing
```
**Action**: Reduce `--gpu-memory-utilization` or batch size

### Thermal Throttling
```
🔥 THERMAL THROTTLING
Bottleneck: thermal
Recommendations:
  1. Check GPU cooling and airflow
  2. Reduce batch size to lower power consumption
```
**Action**: Check cooling, reduce workload intensity

### Low Utilization
```
😴 LOW GPU UTILIZATION
Bottleneck: contention
Recommendations:
  1. Increase batch size to improve GPU utilization
  2. Implement dynamic batching
```
**Action**: Increase batch size or use dynamic batching

### Power Throttling
```
⚡ POWER THROTTLING
Bottleneck: thermal
Recommendations:
  1. Reduce batch size or model precision
  2. Use mixed precision (FP16)
```
**Action**: Use FP16 precision, reduce batch size

## 🛠️ Advanced Usage

### Custom Monitoring Interval
```bash
# Very frequent monitoring (1 second)
python3 llm_monitor_demo.py --interval 1.0

# Less frequent monitoring (10 seconds)
python3 llm_monitor_demo.py --interval 10.0
```

### Integration with Scripts
```bash
# Run in background and log to file
python3 llm_monitor_demo.py --compact > gpu_monitor.log 2>&1 &

# Monitor for specific duration
timeout 300 python3 llm_monitor_demo.py --compact
```

### Stopping the Monitor
- Press `Ctrl+C` to stop gracefully
- The monitor will show a summary before exiting

## 🔧 Troubleshooting

### "dcgmi command not found"
```bash
# Install DCGM
sudo apt update
sudo apt install datacenter-gpu-manager
```

### "Failed to collect metrics"
- Check if DCGM service is running: `sudo systemctl status dcgm`
- Restart DCGM: `sudo systemctl restart dcgm`
- Check GPU visibility: `nvidia-smi`

### No GPU detected
- Verify GPU is visible: `nvidia-smi`
- Check DCGM can see GPU: `dcgmi discovery -l`

## 💡 Tips for vLLM Optimization

1. **Start with monitoring** - Run the monitor first to see baseline
2. **Watch for patterns** - Look for recurring bottlenecks
3. **Adjust gradually** - Make small changes and monitor impact
4. **Use recommendations** - Follow the specific suggestions provided
5. **Monitor continuously** - Keep running during development and testing

## 📚 Example Workflow

```bash
# Terminal 1: Start monitoring
python3 llm_monitor_demo.py --compact

# Terminal 2: Start vLLM with conservative settings
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048

# Terminal 3: Test with requests
curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "microsoft/DialoGPT-medium", "prompt": "Hello, how are you?", "max_tokens": 50}'
```

The monitor will show you exactly what's happening with your GPU and provide specific recommendations for optimization!
