"""
Hardware information collection for benchmark reports.
"""

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp


@dataclass
class HardwareInfo:
    """Hardware and system information for benchmark context"""
    
    # System information
    system: str
    platform: str
    architecture: str
    processor: str
    python_version: str
    
    # Memory information
    total_memory_gb: Optional[float]
    available_memory_gb: Optional[float]
    
    # JAX information
    jax_version: str
    jax_backend: str
    jax_devices: list
    jax_device_count: int
    
    # CPU information
    cpu_cores: Optional[int]
    cpu_threads: Optional[int]
    cpu_frequency_mhz: Optional[float]
    
    # Additional info
    extra_info: Dict[str, Any]


def get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information"""
    cpu_info = {}
    
    try:
        # Get CPU count
        import os
        cpu_info['cores'] = os.cpu_count()
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                
                # Extract processor name
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        cpu_info['model'] = line.split(':')[1].strip()
                        break
                
                # Count physical cores and threads
                cpu_info['threads'] = cpuinfo.count('processor')
                
                # Get frequency info
                for line in cpuinfo.split('\n'):
                    if 'cpu MHz' in line:
                        cpu_info['frequency_mhz'] = float(line.split(':')[1].strip())
                        break
                        
            except Exception:
                pass
                
        # Fallback to platform.processor()
        if 'model' not in cpu_info:
            cpu_info['model'] = platform.processor() or "Unknown"
            
    except Exception:
        cpu_info = {'cores': None, 'model': 'Unknown'}
    
    return cpu_info


def get_memory_info() -> Dict[str, Optional[float]]:
    """Get system memory information"""
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            total_kb = None
            available_kb = None
            
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    total_kb = int(line.split()[1])
                elif 'MemAvailable:' in line:
                    available_kb = int(line.split()[1])
            
            return {
                'total_gb': total_kb / 1024 / 1024 if total_kb else None,
                'available_gb': available_kb / 1024 / 1024 if available_kb else None
            }
        else:
            # For other systems, try psutil if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'total_gb': memory.total / 1024 / 1024 / 1024,
                    'available_gb': memory.available / 1024 / 1024 / 1024
                }
            except ImportError:
                pass
    except Exception:
        pass
    
    return {'total_gb': None, 'available_gb': None}


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available"""
    gpu_info = {'has_gpu': False, 'gpus': []}
    
    try:
        # Try nvidia-smi for NVIDIA GPUs
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            gpu_info['has_gpu'] = True
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_info['gpus'].append({
                            'name': parts[0].strip(),
                            'memory_mb': int(parts[1].strip())
                        })
    except Exception:
        pass
    
    return gpu_info


def get_jax_info() -> Dict[str, Any]:
    """Get JAX-specific information"""
    try:
        devices = jax.devices()
        return {
            'version': jax.__version__,
            'backend': jax.default_backend(),
            'devices': [str(device) for device in devices],
            'device_count': len(devices),
            'device_types': list(set(device.device_kind for device in devices))
        }
    except Exception as e:
        return {
            'version': 'Unknown',
            'backend': 'Unknown', 
            'devices': [],
            'device_count': 0,
            'error': str(e)
        }


def get_hardware_info() -> HardwareInfo:
    """
    Collect comprehensive hardware and system information.
    
    Returns:
        HardwareInfo object with system specifications
    """
    
    # Basic system info
    system_info = {
        'system': platform.system(),
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'python_version': sys.version.split()[0]
    }
    
    # CPU info
    cpu_info = get_cpu_info()
    
    # Memory info
    memory_info = get_memory_info()
    
    # JAX info
    jax_info = get_jax_info()
    
    # GPU info
    gpu_info = get_gpu_info()
    
    # Combine extra information
    extra_info = {
        'gpu_info': gpu_info,
        'platform_details': {
            'machine': platform.machine(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version()
        }
    }
    
    return HardwareInfo(
        system=system_info['system'],
        platform=system_info['platform'],
        architecture=system_info['architecture'],
        processor=cpu_info.get('model', 'Unknown'),
        python_version=system_info['python_version'],
        
        total_memory_gb=memory_info.get('total_gb'),
        available_memory_gb=memory_info.get('available_gb'),
        
        jax_version=jax_info['version'],
        jax_backend=jax_info['backend'],
        jax_devices=jax_info['devices'],
        jax_device_count=jax_info['device_count'],
        
        cpu_cores=cpu_info.get('cores'),
        cpu_threads=cpu_info.get('threads'),
        cpu_frequency_mhz=cpu_info.get('frequency_mhz'),
        
        extra_info=extra_info
    )


def print_hardware_info(hw_info: HardwareInfo) -> None:
    """Print hardware information in a formatted way"""
    
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    
    print(f"System: {hw_info.system}")
    print(f"Platform: {hw_info.platform}")
    print(f"Architecture: {hw_info.architecture}")
    print(f"Processor: {hw_info.processor}")
    print(f"Python Version: {hw_info.python_version}")
    
    if hw_info.cpu_cores:
        print(f"CPU Cores: {hw_info.cpu_cores}")
    if hw_info.cpu_threads:
        print(f"CPU Threads: {hw_info.cpu_threads}")
    if hw_info.cpu_frequency_mhz:
        print(f"CPU Frequency: {hw_info.cpu_frequency_mhz:.0f} MHz")
    
    if hw_info.total_memory_gb:
        print(f"Total Memory: {hw_info.total_memory_gb:.1f} GB")
    if hw_info.available_memory_gb:
        print(f"Available Memory: {hw_info.available_memory_gb:.1f} GB")
    
    print(f"\nJAX Version: {hw_info.jax_version}")
    print(f"JAX Backend: {hw_info.jax_backend}")
    print(f"JAX Devices: {hw_info.jax_device_count}")
    for device in hw_info.jax_devices:
        print(f"  - {device}")
    
    # GPU information
    gpu_info = hw_info.extra_info.get('gpu_info', {})
    if gpu_info.get('has_gpu'):
        print(f"\nGPU Information:")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory_mb']} MB)")
    else:
        print(f"\nGPU: Not detected")
    
    print()