#!/usr/bin/env python3
"""
DCGM (Data Center GPU Manager) Example Script
This script demonstrates basic DCGM functionality using the Python bindings.
"""

import sys
import os

# Add DCGM Python bindings to the path
sys.path.insert(0, '/usr/share/datacenter-gpu-manager-4/bindings/python3')

try:
    import pydcgm
    import dcgm_structs
    import dcgm_fields
    from dcgm_structs import dcgmExceptionClass
    print("✓ Successfully imported DCGM Python bindings")
except ImportError as e:
    print(f"✗ Failed to import DCGM bindings: {e}")
    sys.exit(1)

def main():
    """Main function to demonstrate DCGM functionality."""
    print("DCGM Python Bindings Test")
    print("=" * 30)
    
    try:
        # Initialize DCGM
        dcgm_handle = pydcgm.DcgmHandle()
        print("✓ DCGM handle created successfully")
        
        # Get system interface
        system = dcgm_handle.GetSystem()
        print("✓ DCGM system interface obtained")
        
        # Get GPU discovery interface
        discovery = system.discovery
        print("✓ DCGM discovery interface obtained")
        
        # Get all GPU IDs
        gpu_ids = discovery.GetAllSupportedGpuIds()
        print(f"✓ Found {len(gpu_ids)} supported GPU(s): {gpu_ids}")
        
        if len(gpu_ids) > 0:
            # Get GPU information for each GPU
            for gpu_id in gpu_ids:
                try:
                    gpu_attrs = discovery.GetGpuAttributes(gpu_id)
                    device_name = gpu_attrs.identifiers.deviceName
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode('utf-8')
                    print(f"  GPU {gpu_id}: {device_name}")
                except Exception as e:
                    print(f"  GPU {gpu_id}: Could not get attributes - {e}")
        
        # Test field group creation
        field_group = pydcgm.DcgmFieldGroup(dcgm_handle, "test_group", [dcgm_fields.DCGM_FI_DEV_GPU_TEMP])
        print("✓ Field group created successfully")
        
        print("\n✓ All DCGM operations completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during DCGM operations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
