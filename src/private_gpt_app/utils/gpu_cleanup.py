"""GPU cleanup utilities to prevent memory leaks."""

import subprocess
import os
import signal


def kill_stale_vllm_processes():
    """Kill any stale vLLM processes from previous runs."""
    try:
        # Find vLLM processes
        result = subprocess.run(
            ['pgrep', '-f', 'VLLM::EngineCore'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed = []
            
            for pid in pids:
                try:
                    pid_int = int(pid)
                    # Check if it's not our current process
                    if pid_int != os.getpid():
                        os.kill(pid_int, signal.SIGKILL)
                        killed.append(pid)
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
            
            if killed:
                print(f"🧹 Killed {len(killed)} stale vLLM process(es): {', '.join(killed)}")
                return True
        
        return False
    
    except FileNotFoundError:
        # pgrep not available, skip
        return False
    except Exception as e:
        print(f"⚠️ Error checking for stale processes: {e}")
        return False


def cleanup_gpu_memory():
    """Force cleanup of GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            print("✓ GPU memory cleaned")
    except Exception as e:
        print(f"⚠️ GPU cleanup warning: {e}")
