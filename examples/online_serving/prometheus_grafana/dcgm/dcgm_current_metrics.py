#!/usr/bin/env python3
"""
DCGM Current Metrics Script
This script outputs current GPU metrics using DCGM.
"""

import sys
import os
import subprocess
import json

def run_dcgmi_command(cmd):
    """Run a dcgmi command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main function to show current DCGM metrics."""
    print("DCGM Current Metrics")
    print("=" * 30)
    
    # Check if dcgmi is available
    dcgmi_check = run_dcgmi_command("which dcgmi")
    if "dcgmi" not in dcgmi_check:
        print("✗ dcgmi command not found. Please install DCGM.")
        return 1
    
    print("✓ dcgmi command found")
    
    # Get GPU information
    print("\n--- GPU Information ---")
    gpu_info = run_dcgmi_command("dcgmi discovery -l")
    print(gpu_info)
    
    # Get current metrics using field IDs
    print("\n--- Current GPU Metrics ---")
    
    # Get comprehensive metrics using field IDs
    # Field IDs: 150=GPU Temp, 155=Power, 203=GPU Util, 204=Mem Util, 100=SM Clock, 101=Mem Clock, 250=Mem Total, 251=Mem Free, 252=Mem Used
    comprehensive = run_dcgmi_command("dcgmi dmon -e 150,155,203,204,100,101,250,251,252 -c 1 -d 1000")
    if comprehensive and "Error" not in comprehensive:
        print(comprehensive)
        
        # Parse and display formatted metrics
        lines = comprehensive.strip().split('\n')
        if len(lines) >= 3:  # Header + separator + data
            data_line = lines[2]  # Skip header and separator
            parts = data_line.split()
            if len(parts) >= 10:
                print(f"\n📊 FORMATTED METRICS:")
                print(f"  🌡️  GPU Temperature: {parts[1]}°C")
                print(f"  ⚡ Power Usage: {parts[2]}W")
                print(f"  📈 GPU Utilization: {parts[3]}%")
                print(f"  📈 Memory Utilization: {parts[4]}%")
                print(f"  🕐 SM Clock: {parts[5]}MHz")
                print(f"  🕐 Memory Clock: {parts[6]}MHz")
                print(f"  💾 Memory Total: {parts[7]}MB")
                print(f"  💾 Memory Free: {parts[8]}MB")
                print(f"  💾 Memory Used: {parts[9]}MB")
    else:
        print("  Comprehensive metrics not available")
    
    # Get additional metrics
    print(f"\n--- Additional Metrics ---")
    
    # Temperature and power limits
    temp_power = run_dcgmi_command("dcgmi dmon -e 150,151,152,155,160,161,162 -c 1 -d 1000")
    if temp_power and "Error" not in temp_power:
        print(f"\n🌡️  TEMPERATURE & POWER LIMITS:")
        print(temp_power)
    
    # Clock information
    clocks = run_dcgmi_command("dcgmi dmon -e 100,101,102,110,111,113,114,115 -c 1 -d 1000")
    if clocks and "Error" not in clocks:
        print(f"\n🕐 CLOCK INFORMATION:")
        print(clocks)
    
    # Memory details
    memory = run_dcgmi_command("dcgmi dmon -e 250,251,252,253 -c 1 -d 1000")
    if memory and "Error" not in memory:
        print(f"\n💾 MEMORY DETAILS:")
        print(memory)
    
    # ECC errors
    ecc = run_dcgmi_command("dcgmi dmon -e 310,311,312,313 -c 1 -d 1000")
    if ecc and "Error" not in ecc:
        print(f"\n🔍 ECC ERRORS:")
        print(ecc)
    
    # Show available field IDs
    print(f"\n--- Available DCGM Field IDs ---")
    print("Use 'dcgmi dmon -h' to see all available field IDs")
    print("Common field IDs:")
    print("  1 = Temperature (GPU, Memory)")
    print("  2 = Power (Usage, Limits)")
    print("  3 = Utilization (GPU, Memory, Encoder, Decoder)")
    print("  4 = Clocks (SM, Memory, Video)")
    print("  5 = Memory (Used, Free, Total)")
    print("  6 = ECC Errors")
    print("  7 = PCIe Throughput")
    print("  8 = NVLINK Throughput")
    print("  9 = Violations")
    print("  10 = XID Errors")
    
    print(f"\n✓ DCGM metrics collection completed!")
    print("✓ Use 'dcgmi dmon -s <field_ids> -c <count> -d <delay>' for custom monitoring")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
