"""
BlockSwap - Memory-efficient training for low VRAM systems

This module provides automatic block swapping capabilities to enable training of large models
on systems with limited VRAM by intelligently moving model blocks between GPU and CPU memory.
"""

import gc
import time
import threading
from typing import Dict, List, Optional, Set, Union, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


@dataclass
class BlockInfo:
    """Information about a swappable block."""
    name: str
    module: nn.Module
    device: torch.device
    last_used: float = field(default_factory=time.time)
    memory_size: int = 0
    is_active: bool = False
    dependencies: Set[str] = field(default_factory=set)
    hook_handles: List[RemovableHandle] = field(default_factory=list)
    swap_priority: int = 0  # Higher priority = swap first


class BlockSwapManager:
    """
    Manages automatic swapping of model blocks between GPU and CPU memory.
    
    This system works by:
    1. Monitoring GPU memory usage
    2. Identifying which blocks are currently needed
    3. Swapping inactive blocks to CPU when memory is low
    4. Pre-loading blocks that will be needed soon
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_threshold: float = 0.85,  # Swap when GPU memory usage exceeds this
        max_blocks_on_gpu: Optional[int] = None,  # Maximum blocks to keep on GPU
        enable_async_swap: bool = True,  # Enable asynchronous swapping
        enable_predictive_loading: bool = True,  # Enable predictive block loading
        swap_delay: float = 0.1,  # Minimum time between swaps (seconds)
        debug: bool = False
    ):
        self.model = model
        self.memory_threshold = memory_threshold
        self.max_blocks_on_gpu = max_blocks_on_gpu
        self.enable_async_swap = enable_async_swap
        self.enable_predictive_loading = enable_predictive_loading
        self.swap_delay = swap_delay
        self.debug = debug
        
        # Internal state
        self.blocks: Dict[str, BlockInfo] = {}
        self.active_blocks: Set[str] = set()
        self.swap_queue = deque()
        self.load_queue = deque()
        self.last_swap_time = 0
        self.swap_lock = threading.Lock()
        self.is_enabled = False
        
        # Statistics
        self.stats = {
            'swaps_to_cpu': 0,
            'swaps_to_gpu': 0,
            'memory_saved': 0,
            'swap_time_total': 0,
        }
        
        # Async worker thread
        self.worker_thread = None
        self.stop_worker = threading.Event()
        
        self._identify_swappable_blocks()
        
    def _identify_swappable_blocks(self):
        """Identify which modules can be swapped."""
        swappable_patterns = [
            'transformer_blocks',
            'single_transformer_blocks', 
            'down_blocks',
            'up_blocks',
            'mid_block',
            'decoder',
            'encoder',
            'layers',
            'blocks'
        ]
        
        def is_swappable(name: str, module: nn.Module) -> bool:
            # Skip modules without parameters
            param_count = sum(p.numel() for p in module.parameters())
            if param_count == 0:
                return False
            
            # Check if module matches swappable patterns
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in swappable_patterns):
                return True
            
            # Check if module has significant parameters (1M+ parameters)
            if param_count > 1000000:
                return True
                
            return False
        
        for name, module in self.model.named_modules():
            if is_swappable(name, module):
                # Skip modules without parameters
                params = list(module.parameters())
                if not params:
                    continue
                    
                # Calculate memory size
                memory_size = sum(
                    p.numel() * p.element_size() 
                    for p in params
                )
                
                # Get device from first parameter
                device = params[0].device
                
                block_info = BlockInfo(
                    name=name,
                    module=module,
                    device=device,
                    memory_size=memory_size
                )
                
                self.blocks[name] = block_info
                
                if self.debug:
                    print(f"BlockSwap: Identified swappable block '{name}' "
                          f"({memory_size / 1024**2:.1f} MB)")
    
    def enable(self):
        """Enable blockswapping."""
        if self.is_enabled:
            return
            
        self.is_enabled = True
        self._install_hooks()
        
        if self.enable_async_swap and self.worker_thread is None:
            self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
            self.worker_thread.start()
            
        if self.debug:
            print(f"BlockSwap: Enabled with {len(self.blocks)} swappable blocks")
    
    def disable(self):
        """Disable blockswapping."""
        if not self.is_enabled:
            return
            
        self.is_enabled = False
        self._remove_hooks()
        
        if self.worker_thread is not None:
            self.stop_worker.set()
            self.worker_thread.join(timeout=5.0)
            self.worker_thread = None
            self.stop_worker.clear()
        
        # Move all blocks back to original device
        for block_info in self.blocks.values():
            try:
                current_device = self._get_module_device(block_info.module)
                if current_device and block_info.device != current_device:
                    block_info.module.to(block_info.device)
            except Exception as e:
                if self.debug:
                    print(f"BlockSwap: Error restoring device for block: {e}")
        
        if self.debug:
            print("BlockSwap: Disabled")
    
    def _install_hooks(self):
        """Install forward hooks to track block usage."""
        for name, block_info in self.blocks.items():
            # Pre-forward hook
            def make_pre_hook(block_name):
                def pre_hook(module, input):
                    self._on_block_access(block_name)
                return pre_hook
            
            # Post-forward hook  
            def make_post_hook(block_name):
                def post_hook(module, input, output):
                    self._on_block_complete(block_name)
                return post_hook
            
            pre_handle = block_info.module.register_forward_pre_hook(make_pre_hook(name))
            post_handle = block_info.module.register_forward_hook(make_post_hook(name))
            
            block_info.hook_handles.extend([pre_handle, post_handle])
    
    def _remove_hooks(self):
        """Remove all installed hooks."""
        for block_info in self.blocks.values():
            for handle in block_info.hook_handles:
                handle.remove()
            block_info.hook_handles.clear()
    
    def _on_block_access(self, block_name: str):
        """Called when a block is about to be accessed."""
        if not self.is_enabled:
            return
            
        block_info = self.blocks.get(block_name)
        if not block_info:
            return
        
        # Ensure block is on GPU
        if self._get_module_device_type(block_info.module) == 'cpu':
            self._swap_to_gpu(block_name, urgent=True)
        
        # Mark as active
        block_info.is_active = True
        block_info.last_used = time.time()
        self.active_blocks.add(block_name)
        
        # Check if we need to swap other blocks
        self._check_memory_pressure()
    
    def _on_block_complete(self, block_name: str):
        """Called when a block completes forward pass."""
        if not self.is_enabled:
            return
            
        block_info = self.blocks.get(block_name)
        if not block_info:
            return
        
        # Mark as inactive after a short delay
        def delayed_deactivate():
            time.sleep(0.05)  # Small delay to handle immediate reuse
            block_info.is_active = False
            self.active_blocks.discard(block_name)
        
        if self.enable_async_swap:
            threading.Thread(target=delayed_deactivate, daemon=True).start()
        else:
            delayed_deactivate()
    
    def _get_module_device(self, module: nn.Module) -> Optional[torch.device]:
        """Get the device of a module, with fallback methods."""
        try:
            # Method 1: Direct device attribute
            if hasattr(module, 'device'):
                return module.device
            
            # Method 2: Weight device
            if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                return module.weight.device
            
            # Method 3: First parameter device
            params = list(module.parameters())
            if params:
                return params[0].device
            
            # Method 4: First buffer device
            buffers = list(module.buffers())
            if buffers:
                return buffers[0].device
                
        except (AttributeError, StopIteration):
            pass
        
        return None
    
    def _get_module_device_type(self, module: nn.Module) -> str:
        """Get the device type of a module ('cuda' or 'cpu')."""
        device = self._get_module_device(module)
        return device.type if device is not None else 'cpu'
    
    def _check_memory_pressure(self):
        """Check if we need to free GPU memory."""
        if not torch.cuda.is_available():
            return
        
        try:
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            
            # Avoid division by zero
            if max_memory == 0:
                return
                
            memory_used = memory_allocated / max_memory
            
            if memory_used > self.memory_threshold:
                self._free_gpu_memory()
            
            # Also check max blocks limit
            if self.max_blocks_on_gpu is not None:
                gpu_blocks = [
                    name for name, block in self.blocks.items() 
                    if self._get_module_device_type(block.module) == 'cuda'
                ]
                
                if len(gpu_blocks) > self.max_blocks_on_gpu:
                    self._free_gpu_memory()
                    
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error checking memory pressure: {e}")
    
    def _free_gpu_memory(self):
        """Free GPU memory by swapping blocks to CPU."""
        with self.swap_lock:
            # Find inactive blocks on GPU
            candidates = [
                (name, block) for name, block in self.blocks.items()
                if (self._get_module_device_type(block.module) == 'cuda' and 
                    not block.is_active and 
                    name not in self.active_blocks)
            ]
            
            # Sort by last used time (oldest first) and priority
            candidates.sort(key=lambda x: (x[1].swap_priority, x[1].last_used))
            
            # Swap oldest/lowest priority blocks
            for name, block in candidates[:max(1, len(candidates) // 2)]:
                self._swap_to_cpu(name)
    
    def _swap_to_cpu(self, block_name: str):
        """Swap a block from GPU to CPU."""
        if time.time() - self.last_swap_time < self.swap_delay:
            return
        
        block_info = self.blocks.get(block_name)
        if not block_info or self._get_module_device_type(block_info.module) == 'cpu':
            return
        
        # Safety check - ensure module has parameters
        if not any(block_info.module.parameters()):
            return
        
        if self.debug:
            print(f"BlockSwap: Moving '{block_name}' to CPU")
        
        start_time = time.time()
        
        try:
            # Move to CPU
            block_info.module.to('cpu', non_blocking=True)
            
            # Update stats
            self.stats['swaps_to_cpu'] += 1
            self.stats['memory_saved'] += block_info.memory_size
            self.stats['swap_time_total'] += time.time() - start_time
            self.last_swap_time = time.time()
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error moving '{block_name}' to CPU: {e}")
    
    def _swap_to_gpu(self, block_name: str, urgent: bool = False):
        """Swap a block from CPU to GPU."""
        if not urgent and time.time() - self.last_swap_time < self.swap_delay:
            return
        
        block_info = self.blocks.get(block_name)
        if not block_info or self._get_module_device_type(block_info.module) == 'cuda':
            return
        
        # Safety check - ensure module has parameters
        if not any(block_info.module.parameters()):
            return
        
        if self.debug:
            print(f"BlockSwap: Moving '{block_name}' to GPU")
        
        start_time = time.time()
        
        try:
            # Ensure we have enough GPU memory
            if not urgent:
                self._check_memory_pressure()
            
            # Move to GPU
            device = block_info.device if block_info.device.type == 'cuda' else 'cuda'
            block_info.module.to(device, non_blocking=True)
            
            # Update stats
            self.stats['swaps_to_gpu'] += 1
            self.stats['swap_time_total'] += time.time() - start_time
            self.last_swap_time = time.time()
            
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error moving '{block_name}' to GPU: {e}")
    
    def _async_worker(self):
        """Async worker thread for non-critical swapping operations."""
        while not self.stop_worker.wait(0.1):  # Check every 100ms
            try:
                # Process swap queue
                while self.swap_queue and not self.stop_worker.is_set():
                    block_name = self.swap_queue.popleft()
                    self._swap_to_cpu(block_name)
                
                # Process load queue
                while self.load_queue and not self.stop_worker.is_set():
                    block_name = self.load_queue.popleft()
                    self._swap_to_gpu(block_name)
                
                # Predictive loading
                if self.enable_predictive_loading:
                    self._predictive_load()
                    
            except Exception as e:
                if self.debug:
                    print(f"BlockSwap worker error: {e}")
    
    def _predictive_load(self):
        """Predictively load blocks that might be needed soon."""
        # This is a simple heuristic - can be made more sophisticated
        # For now, just ensure blocks that were recently active are available
        
        current_time = time.time()
        recently_used_threshold = 2.0  # seconds
        
        for name, block in self.blocks.items():
            if (current_time - block.last_used < recently_used_threshold and
                self._get_module_device_type(block.module) == 'cpu' and
                not block.is_active):
                
                # Add to load queue if not already there
                if name not in self.load_queue:
                    self.load_queue.append(name)
    
    @contextmanager
    def preserve_block(self, block_name: str):
        """Context manager to ensure a block stays on GPU."""
        block_info = self.blocks.get(block_name)
        if not block_info:
            yield
            return
        
        # Temporarily increase priority
        original_priority = block_info.swap_priority
        block_info.swap_priority = 1000  # High priority
        
        # Ensure it's on GPU
        if self._get_module_device_type(block_info.module) == 'cpu':
            self._swap_to_gpu(block_name, urgent=True)
        
        try:
            yield
        finally:
            # Restore original priority
            block_info.swap_priority = original_priority
    
    def get_stats(self) -> Dict[str, Any]:
        """Get swapping statistics."""
        gpu_blocks = 0
        for block in self.blocks.values():
            try:
                # Try to get device from module
                if hasattr(block.module, 'device'):
                    device_type = block.module.device.type
                elif hasattr(block.module, 'weight') and hasattr(block.module.weight, 'device'):
                    device_type = block.module.weight.device.type
                else:
                    # Fallback: check first parameter
                    params = list(block.module.parameters())
                    if params:
                        device_type = params[0].device.type
                    else:
                        continue  # Skip modules without parameters or device info
                
                if device_type == 'cuda':
                    gpu_blocks += 1
            except AttributeError:
                # Skip modules that don't have device info
                continue
        
        cpu_blocks = len(self.blocks) - gpu_blocks
        
        stats = self.stats.copy()
        stats.update({
            'total_blocks': len(self.blocks),
            'gpu_blocks': gpu_blocks,
            'cpu_blocks': cpu_blocks,
            'active_blocks': len(self.active_blocks),
            'memory_saved_mb': self.stats['memory_saved'] / 1024**2,
            'avg_swap_time': (
                self.stats['swap_time_total'] / max(1, self.stats['swaps_to_cpu'] + self.stats['swaps_to_gpu'])
            )
        })
        
        return stats
    
    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("BlockSwap Statistics")
        print("="*50)
        print(f"Total Blocks: {stats['total_blocks']}")
        print(f"GPU Blocks: {stats['gpu_blocks']}")
        print(f"CPU Blocks: {stats['cpu_blocks']}")
        print(f"Active Blocks: {stats['active_blocks']}")
        print(f"Swaps to CPU: {stats['swaps_to_cpu']}")
        print(f"Swaps to GPU: {stats['swaps_to_gpu']}")
        print(f"Memory Saved: {stats['memory_saved_mb']:.1f} MB")
        print(f"Avg Swap Time: {stats['avg_swap_time']*1000:.1f} ms")
        print("="*50)
    
    def __enter__(self):
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


def create_blockswap_manager(
    model: nn.Module,
    memory_threshold: float = 0.85,
    max_blocks_on_gpu: Optional[int] = None,
    **kwargs
) -> BlockSwapManager:
    """Create and configure a BlockSwapManager for the given model."""
    return BlockSwapManager(
        model=model,
        memory_threshold=memory_threshold,
        max_blocks_on_gpu=max_blocks_on_gpu,
        **kwargs
    )


# Convenience function for integration
def enable_blockswap(
    model: nn.Module,
    memory_threshold: float = 0.85,
    max_blocks_on_gpu: Optional[int] = None,
    **kwargs
) -> BlockSwapManager:
    """
    Enable blockswapping for a model with sensible defaults.
    
    Args:
        model: The model to enable blockswapping for
        memory_threshold: GPU memory threshold (0.85 = 85%)
        max_blocks_on_gpu: Maximum number of blocks to keep on GPU
        **kwargs: Additional arguments for BlockSwapManager
    
    Returns:
        BlockSwapManager instance
    """
    manager = create_blockswap_manager(
        model, memory_threshold, max_blocks_on_gpu, **kwargs
    )
    manager.enable()
    return manager
