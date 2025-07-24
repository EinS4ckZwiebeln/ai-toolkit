import gc
import time
import threading
import re
from typing import Dict, List, Optional, Set
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

    def __post_init__(self):
        """Ensure is_active is always a Python bool."""
        # Convert any tensor values to bool
        if hasattr(self.is_active, 'item'):
            self.is_active = bool(self.is_active.item())
        else:
            self.is_active = bool(self.is_active)


class BlockSwapManager:
    def __init__(
        self,
        model: nn.Module,
        memory_threshold: float = 0.85,  # Swap when GPU memory usage exceeds this
        max_blocks_on_gpu: Optional[int] = None,  # Maximum blocks to keep on GPU
        enable_async_swap: bool = True,  # Enable asynchronous swapping
        enable_predictive_loading: bool = True,  # Enable predictive block loading
        swap_delay: float = 0.025,  # Minimum time between swaps (seconds)
        debug: bool = False,
        use_pinned_memory: bool = True,
        gradient_checkpointing_safe: bool = True  # Enable checkpointing compatibility
    ):
        self.model = model
        self.memory_threshold = memory_threshold
        self.max_blocks_on_gpu = max_blocks_on_gpu
        self.enable_async_swap = enable_async_swap
        self.enable_predictive_loading = enable_predictive_loading
        self.swap_delay = swap_delay
        self.debug = debug
        self.use_pinned_memory = use_pinned_memory
        self.gradient_checkpointing_safe = gradient_checkpointing_safe

        # Internal state
        self.blocks: Dict[str, BlockInfo] = {}
        self.active_blocks: Set[str] = set()
        self.swap_queue = deque()
        self.load_queue = deque()
        self.last_swap_time = 0
        self.swap_lock = threading.Lock()
        self.is_enabled = False

        # Gradient checkpointing compatibility
        self._in_backward_pass = False
        self._backward_context_depth = 0
        self._preserved_blocks: Set[str] = set()
        self._backward_hooks_installed = False # <--- ADDED: Initialize the flag
        
        # Async worker thread
        self.worker_thread: Optional[threading.Thread] = None # <--- ADDED: Initialize worker_thread
        self.stop_worker = threading.Event()

        self._identify_swappable_blocks()
        self._install_backward_hooks() # This will now safely use _backward_hooks_installed

    def _install_backward_hooks(self):
        """Install hooks to detect backward pass for gradient checkpointing compatibility."""
        if not self.gradient_checkpointing_safe:
            return

        # Only install hooks if they haven't been installed yet
        if self._backward_hooks_installed:
            return

        def pre_backward_hook(module, grad_output):
            """Called before backward pass starts."""
            self._in_backward_pass = True
            self._backward_context_depth += 1

        def post_backward_hook(module, grad_input, grad_output):
            """Called after backward pass completes."""
            self._backward_context_depth -= 1
            if self._backward_context_depth <= 0:
                self._in_backward_pass = False
                self._backward_context_depth = 0
                # After backward pass, we can safely swap again
                if self.is_enabled:
                    self._check_memory_pressure()

        # Install hooks on the model to detect backward pass
        # We install on the first module with parameters found to avoid multiple hooks
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:  # Only modules with parameters
                module.register_full_backward_pre_hook(pre_backward_hook)
                module.register_full_backward_hook(post_backward_hook)
                self._backward_hooks_installed = True # Mark as installed
                if self.debug:
                    print(f"BlockSwap: Installed backward hooks on module: {name}")
                break  # Only need to install on one module

    def _identify_swappable_blocks(self):
        """Identify which modules can be swapped with better granularity."""
        # More restrictive patterns - only safe sub-modules that won't interfere with gradient checkpointing
        swappable_patterns = [
            'transformer_blocks\\.\d+\\.ff\\.net\\.\d+$',    # Only FF net sub-layers (final layer)
            'single_transformer_blocks\\.\d+\\.ff\\.net\\.\d+$',
            'down_blocks\\.\d+\\.resnets\\.\d+\\.conv\d+$',  # Only conv layers in resnets
            'up_blocks\\.\d+\\.resnets\\.\d+\\.conv\d+$',
        ]

        # Aggressive exclusion for gradient checkpointing compatibility
        exclude_patterns = [
            # Top-level containers and checkpoint boundaries
            'transformer_blocks$',
            'single_transformer_blocks$',
            'model$',
            'unet$',
            'text_encoder$',
            'vae$',

            # Attention components (critical for gradient checkpointing)
            '.*\\.attn.*',
            '.*\\.attention.*',

            # Any layers involved in gradient computation
            '.*\\.norm.*',
            '.*\\.ln.*',
            '.*\\.layer_norm.*',

            # Critical output layers
            'norm_out',
            'proj_out',
            'to_out',
            'final_layer',
            'output_projection',

            # Quantized components
            '.*\\.bias$',
            '.*\\.qweight$',
            '.*\\.qbias$',
            '.*\\.qscale$',
            '.*\\.qzero$',
            '.*quant.*',
            '.*qlinear.*',

            # Linear layers and projections
            '.*\\.linear$',
            '.*\\.proj$',
            '.*\\.fc\\d*$',
            '.*\\.to_q$',
            '.*\\.to_k$',
            '.*\\.to_v$',
            '.*\\.to_out$',
        ]

        def is_swappable(name: str, module: nn.Module) -> bool:
            # Skip modules without parameters
            param_count = sum(p.numel() for p in module.parameters())
            if param_count == 0:
                return False

            # Skip very small modules (< 1MB)
            if param_count < 1024 * 1024:
                return False

            # Skip if module name contains any exclude pattern
            for pattern in exclude_patterns:
                if re.search(pattern, name):
                    return False

            # Skip if module has any quantized parameters
            for param in module.parameters():
                # Check the parameter's name or type for quantization
                # Note: `param.name` is not a standard PyTorch parameter attribute.
                # Usually, one checks the parent module's type or specific parameter naming conventions.
                # For robustness, we check the module name itself for quant indicators.
                if any(exclude in name.lower() for exclude in ['qweight', 'qbias', 'quant', 'qlinear']):
                    return False

            # Check if module matches swappable patterns
            for pattern in swappable_patterns:
                if re.search(pattern, name):
                    return True

            return False

        for name, module in self.model.named_modules():
            if is_swappable(name, module):
                # Get all parameters
                params = list(module.parameters())
                if not params:
                    continue

                # Calculate memory size
                memory_size = sum(
                    p.numel() * p.element_size()
                    for p in params
                )

                # Skip very small blocks
                if memory_size < 1024 * 1024:
                    continue

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
                    print(f"BlockSwap: Identified swappable block: {name} ({memory_size / 1024**2:.1f} MB)")

    def enable(self):
        """Enable blockswapping."""
        if self.is_enabled:
            return

        self.is_enabled = True
        self._install_hooks()

        if self.enable_async_swap and self.worker_thread is None:
            self.stop_worker.clear() # Ensure event is clear before starting
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
            if self.worker_thread.is_alive():
                if self.debug:
                    print("BlockSwap: Warning - Async worker thread did not stop gracefully.")
            self.worker_thread = None
            self.stop_worker.clear() # Reset event for future enables

        # Move all blocks back to original device
        for block_info in self.blocks.values():
            try:
                current_device = self._get_module_device(block_info.module)
                # Only move if currently on CPU and original device was CUDA
                if current_device and current_device.type == 'cpu' and block_info.device.type == 'cuda':
                    if self.debug:
                        print(f"BlockSwap: Moving '{block_info.name}' back to its original device: {block_info.device}")
                    block_info.module.to(block_info.device)
            except Exception as e:
                if self.debug:
                    print(f"BlockSwap: Error restoring device for block '{block_info.name}': {e}")
        
        # Clear queues after disabling and moving blocks back
        self.swap_queue.clear()
        self.load_queue.clear()
        
        # Explicitly clear CUDA cache and collect garbage one last time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if self.debug:
            print("BlockSwap: Disabled")

    def _install_hooks(self):
        """Install forward hooks to track block usage. Ensures no duplicate hooks."""
        self._remove_hooks() # First, remove any existing hooks to prevent duplicates

        for name, block_info in self.blocks.items():
            # Pre-forward hook
            def make_pre_hook(block_name_): # Use a different variable name for closure
                def pre_hook(module, input):
                    self._on_block_access(block_name_)
                return pre_hook

            # Post-forward hook
            def make_post_hook(block_name_): # Use a different variable name for closure
                def post_hook(module, input, output):
                    self._on_block_complete(block_name_)
                return post_hook

            pre_handle = block_info.module.register_forward_pre_hook(make_pre_hook(name))
            post_handle = block_info.module.register_forward_hook(make_post_hook(name))

            block_info.hook_handles.extend([pre_handle, post_handle])

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for block_info in self.blocks.values():
            for handle in block_info.hook_handles:
                try:
                    handle.remove()
                except RuntimeError as e:
                    if self.debug:
                        print(f"BlockSwap: Error removing hook for '{block_info.name}': {e}")
            block_info.hook_handles.clear() # Always clear the list after attempting removal

    def _on_block_access(self, block_name: str):
        """Called when a block is about to be accessed."""
        if not self.is_enabled:
            return

        block_info = self.blocks.get(block_name)
        if not block_info:
            return

        # Ensure block is on GPU if it's currently on CPU
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
            if self.is_enabled: # Only deactivate if manager is still enabled
                block_info.is_active = False
                self.active_blocks.discard(block_name)

        if self.enable_async_swap:
            # Use a non-daemon thread if you want to ensure it completes even if main thread exits
            # However, for cleanup, a daemon thread is usually fine.
            threading.Thread(target=delayed_deactivate, daemon=True).start()
        else:
            delayed_deactivate()

    def _get_module_device(self, module: nn.Module) -> Optional[torch.device]:
        """Get the device of a module, with fallback methods."""
        try:
            # Method 1: Direct device attribute (if module has it, e.g., FSDP)
            if hasattr(module, 'device'):
                return module.device

            # Method 2: First parameter device
            params = list(module.parameters())
            if params:
                return params[0].device

            # Method 3: First buffer device
            buffers = list(module.buffers())
            if buffers:
                return buffers[0].device

        except (AttributeError, StopIteration, RuntimeError):
            # RuntimeError can happen if a tensor is on an invalid device
            pass

        return None

    def _get_module_device_type(self, module: nn.Module) -> str:
        """Get the device type of a module ('cuda' or 'cpu')."""
        device = self._get_module_device(module)
        return str(device.type) if device is not None else 'cpu'

    def _check_memory_pressure(self):
        """Check if we need to free GPU memory."""
        if not torch.cuda.is_available():
            return
        # Do not swap during backward pass if gradient checkpointing safe is enabled
        if self.gradient_checkpointing_safe and self._in_backward_pass:
            if self.debug:
                print("BlockSwap: Skipping memory check during backward pass (gradient checkpointing safe).")
            return
        try:
            device = torch.device("cuda:0")
            memory_allocated = torch.cuda.memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory

            # Convert to int if needed, using .item() if it's a tensor
            if isinstance(memory_allocated, torch.Tensor):
                memory_allocated = memory_allocated.item()
            if isinstance(total_memory, torch.Tensor):
                total_memory = total_memory.item()
            memory_allocated = int(memory_allocated)
            total_memory = int(total_memory)

            if total_memory == 0:
                return
            memory_used = float(memory_allocated) / float(total_memory)

            # Ensure memory_threshold is a float
            memory_threshold = float(self.memory_threshold)

            pressure_active = False
            if float(memory_used) > float(memory_threshold):
                if self.debug:
                    print(f"BlockSwap: Memory pressure detected - {memory_used:.2f} > {memory_threshold}")
                pressure_active = True

            # Also check max blocks limit
            gpu_blocks_count = 0
            for name, block in self.blocks.items():
                try:
                    device_type = self._get_module_device_type(block.module)
                    if str(device_type) == 'cuda':
                        gpu_blocks_count += 1
                except Exception as e:
                    if self.debug:
                        print(f"BlockSwap: Error checking device type for block '{name}': {e}")

            if self.max_blocks_on_gpu is not None and gpu_blocks_count > self.max_blocks_on_gpu:
                if self.debug:
                    print(f"BlockSwap: Max blocks limit exceeded - {gpu_blocks_count} > {self.max_blocks_on_gpu}")
                pressure_active = True

            if pressure_active:
                self._free_gpu_memory()

        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error checking memory pressure: {e}")

    def _free_gpu_memory(self):
        """Free GPU memory by swapping blocks to CPU with better memory management."""
        # Prevent swapping during backward pass if gradient checkpointing is enabled
        if self.gradient_checkpointing_safe and self._in_backward_pass:
            if self.debug:
                print("BlockSwap: Skipping _free_gpu_memory during backward pass (gradient checkpointing safe).")
            return
        
        with self.swap_lock:
            if self.debug:
                print("BlockSwap: Starting _free_gpu_memory")

            # Find inactive blocks on GPU
            candidates = []
            for name, block in self.blocks.items():
                try:
                    device_type = self._get_module_device_type(block.module)

                    # Ensure is_active is a bool
                    is_active_bool = bool(block.is_active)
                    
                    # Ensure name is not in active_blocks set
                    in_active_set = name in self.active_blocks

                    # Don't swap if active, or if it's a preserved block
                    if str(device_type) == 'cuda' and not is_active_bool and not in_active_set and name not in self._preserved_blocks:
                        candidates.append((name, block))
                        if self.debug:
                            print(f"BlockSwap: [free_gpu_memory] Added '{name}' to candidates")

                except Exception as e:
                    if self.debug:
                        print(f"BlockSwap: Error processing block '{name}' in _free_gpu_memory: {e}")

            if self.debug:
                print(f"BlockSwap: Found {len(candidates)} candidate blocks for swapping")

            if not candidates:
                if self.debug:
                    print("BlockSwap: No suitable candidates for swapping.")
                return

            # Sort by memory size (largest first) to free memory more quickly, then by last used (LRU)
            candidates.sort(key=lambda x: (x[1].swap_priority, x[1].memory_size, x[1].last_used), reverse=True)

            # Calculate how much memory we need to free
            memory_to_free = 0 # Initialize to ensure it's defined
            try:
                device = torch.device("cuda:0")
                memory_allocated = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                memory_threshold = float(self.memory_threshold)

                target_memory = total_memory * memory_threshold
                memory_to_free = memory_allocated - target_memory

                # If memory_to_free is negative, it means we are below the threshold, so no need to free.
                if memory_to_free <= 0:
                    if self.debug:
                        print(f"BlockSwap: Current GPU memory ({memory_allocated / 1024**2:.1f} MB) is already below or at target ({target_memory / 1024**2:.1f} MB). No swap needed.")
                    return

                if self.debug:
                    print(f"BlockSwap: Need to free {memory_to_free / 1024**2:.1f} MB")
                    print(f"BlockSwap: Current usage: {memory_allocated / 1024**2:.1f} MB")
                    print(f"BlockSwap: Target usage: {target_memory / 1024**2:.1f} MB")

            except Exception as e:
                if self.debug:
                    print(f"BlockSwap: Error calculating memory needs: {e}. Defaulting to freeing 10% of total memory.")
                # Fallback: if cannot get current memory, try to free 10% of theoretical total
                try:
                    device = torch.device("cuda:0")
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    memory_to_free = total_memory * 0.1
                except Exception:
                    memory_to_free = 100 * 1024 * 1024 # Default to 100MB if all fails

            # Select blocks to swap based on memory size and a minimum of 5 blocks
            blocks_to_swap = []
            freed_memory = 0

            for name, block in candidates:
                if freed_memory >= memory_to_free and len(blocks_to_swap) >= 5:
                    break
                blocks_to_swap.append((name, block))
                freed_memory += block.memory_size

            if self.debug:
                print(f"BlockSwap: Will swap {len(blocks_to_swap)} blocks, targeting to free {freed_memory / 1024**2:.1f} MB")

            # Use very small batch size to avoid memory spikes
            batch_size = max(1, min(3, len(blocks_to_swap)))  # Process in tiny batches

            for i in range(0, len(blocks_to_swap), batch_size):
                batch = blocks_to_swap[i:i+batch_size]

                if self.debug:
                    print(f"BlockSwap: Processing swap batch {i//batch_size + 1}/{(len(blocks_to_swap) + batch_size - 1)//batch_size}")

                for name, block in batch:
                    try:
                        self._swap_to_cpu(name)
                        # Small delay between swaps to allow memory to stabilize
                        if len(batch) > 1:
                            time.sleep(0.01) # Even smaller delay

                    except Exception as e:
                        if self.debug:
                            print(f"BlockSwap: Error swapping '{name}' to CPU: {e}")

                # Force garbage collection and empty cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()  # Force Python garbage collection

                # Check if we've freed enough memory after the batch
                try:
                    current_memory = torch.cuda.memory_allocated(device)
                    if current_memory <= target_memory:
                        if self.debug:
                            print(f"BlockSwap: Memory pressure relieved after {i + len(batch)} swaps. Current: {current_memory / 1024**2:.1f} MB, Target: {target_memory / 1024**2:.1f} MB")
                        break
                except Exception as e:
                    if self.debug:
                        print(f"BlockSwap: Error checking memory after batch: {e}")
                    pass # Continue swapping if check fails

    def _swap_to_cpu(self, block_name: str):
        """Swap a block from GPU to CPU with improved memory handling."""
        # Prevent swapping during backward pass if gradient checkpointing is enabled
        if self.gradient_checkpointing_safe and self._in_backward_pass:
            if self.debug:
                print(f"BlockSwap: Skipping swap_to_cpu('{block_name}') during backward pass (gradient checkpointing safe).")
            return

        if time.time() - self.last_swap_time < self.swap_delay:
            return

        block_info = self.blocks.get(block_name)
        if not block_info:
            return

        if self._get_module_device_type(block_info.module) == 'cpu':
            if self.debug:
                print(f"BlockSwap: Block '{block_name}' already on CPU. Skipping swap.")
            return

        # Safety check - ensure module has parameters
        try:
            param_list = list(block_info.module.parameters())
            if not param_list:  # Check if list is empty
                if self.debug:
                    print(f"BlockSwap: Block '{block_name}' has no parameters to swap. Skipping.")
                return
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error accessing parameters for '{block_name}': {e}. Skipping swap.")
            return

        if self.debug:
            print(f"BlockSwap: Moving '{block_name}' to CPU (size: {block_info.memory_size / 1024**2:.1f} MB, pinned={self.use_pinned_memory})")

        try:
            # Check for UintxTensor in parameters or buffers before any move
            has_uintx = any(
                (p.device.type == 'cuda' and p.__class__.__name__ == 'UintxTensor')
                for p in block_info.module.parameters()
            ) or any(
                (b.device.type == 'cuda' and b.__class__.__name__ == 'UintxTensor')
                for b in block_info.module.buffers()
            )
            if has_uintx:
                if self.debug:
                    print(f"BlockSwap: Skipping all move/pin for '{block_name}' due to UintxTensor (quantized weights)")
                return  # Skip all move/pin for this module

            # Move parameters one by one to reduce memory spikes
            for param in block_info.module.parameters():
                if param.device.type == 'cuda':
                    cpu_param = param.detach().cpu()
                    param.data = cpu_param
            for buf in block_info.module.buffers():
                if buf.device.type == 'cuda':
                    cpu_buf = buf.detach().cpu()
                    buf.data = cpu_buf

            # Pin memory for all parameters and buffers if enabled
            if self.use_pinned_memory:
                try:
                    for param in block_info.module.parameters():
                        if hasattr(param, 'data') and param.data.device.type == 'cpu' and not param.data.is_pinned():
                            param.data = param.data.pin_memory()
                    for buf in block_info.module.buffers():
                        if hasattr(buf, 'data') and buf.data.device.type == 'cpu' and not buf.data.is_pinned():
                            buf.data = buf.data.pin_memory()
                except Exception as e:
                    if self.debug:
                        print(f"BlockSwap: Warning - could not pin memory for '{block_name}': {e}")

            self.last_swap_time = time.time()

            # Force garbage collection and empty cache after each successful swap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if self.debug:
                print(f"BlockSwap: Successfully moved '{block_name}' to CPU")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.debug:
                    print(f"BlockSwap: CUDA OOM while swapping '{block_name}' - trying alternative approach (direct .to('cpu') without pinning attempt)")
                try:
                    # Fallback to direct .to('cpu') without pinning
                    block_info.module.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    self.last_swap_time = time.time()
                    if self.debug:
                        print(f"BlockSwap: Successfully moved '{block_name}' to CPU with fallback.")
                except Exception as e2:
                    if self.debug:
                        print(f"BlockSwap: Failed to swap '{block_name}' even with fallback: {e2}")
            else:
                if self.debug:
                    print(f"BlockSwap: Failed moving '{block_name}' to CPU: {e}")
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Unexpected error moving '{block_name}' to CPU: {e}")

    def _swap_to_gpu(self, block_name: str, urgent: bool = False):
        """Swap a block from CPU (possibly pinned) to GPU."""
        if not urgent and time.time() - self.last_swap_time < self.swap_delay:
            return

        block_info = self.blocks.get(block_name)
        if not block_info:
            return

        if self._get_module_device_type(block_info.module) == 'cuda':
            if self.debug:
                print(f"BlockSwap: Block '{block_name}' already on GPU. Skipping load.")
            return

        # Safety check - ensure module has parameters
        try:
            param_list = list(block_info.module.parameters())
            if not param_list:  # Check if list is empty
                if self.debug:
                    print(f"BlockSwap: Block '{block_name}' has no parameters to load. Skipping.")
                return
        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Error accessing parameters for '{block_name}': {e}. Skipping load.")
            return

        if self.debug:
            print(f"BlockSwap: Moving '{block_name}' to GPU")

        try:
            if not urgent:
                # Check memory pressure before loading, unless it's an urgent request
                self._check_memory_pressure()

            # Fix: set device properly - ensure target device is CUDA.
            # If block_info.device is valid and CUDA, use it. Otherwise, default to cuda:0.
            target_device = block_info.device if block_info.device.type == 'cuda' else torch.device('cuda:0')

            block_info.module.to(target_device, non_blocking=True) # Use non_blocking for async
            torch.cuda.current_stream().synchronize() # Synchronize current stream after async move for correctness

            self.last_swap_time = time.time()

            # Force garbage collection and empty cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            if self.debug:
                print(f"BlockSwap: Failed moving '{block_name}' to GPU: {e}")

    def _async_worker(self):
        """Async worker thread for non-critical swapping operations."""
        while not self.stop_worker.wait(0.1):  # Check every 100ms
            try:
                # Process swap queue
                while self.swap_queue and not self.stop_worker.is_set():
                    block_name = self.swap_queue.popleft()
                    # Only swap if not currently active or preserved
                    block_info = self.blocks.get(block_name)
                    if block_info and not block_info.is_active and block_name not in self._preserved_blocks:
                        self._swap_to_cpu(block_name)
                    elif self.debug:
                        if block_info and (block_info.is_active or block_name in self._preserved_blocks):
                            print(f"BlockSwap worker: Skipping swap of '{block_name}' as it is active or preserved.")
                        else:
                            print(f"BlockSwap worker: Skipping swap of unknown block '{block_name}'.")

                # Process load queue
                while self.load_queue and not self.stop_worker.is_set():
                    block_name = self.load_queue.popleft()
                    # Only load if currently on CPU and not already on GPU
                    block_info = self.blocks.get(block_name)
                    if block_info and self._get_module_device_type(block_info.module) == 'cpu':
                        self._swap_to_gpu(block_name)
                    elif self.debug:
                        if block_info and self._get_module_device_type(block_info.module) == 'cuda':
                            print(f"BlockSwap worker: Skipping load of '{block_name}' as it is already on GPU.")
                        else:
                            print(f"BlockSwap worker: Skipping load of unknown block '{block_name}'.")

                # Predictive loading
                if self.enable_predictive_loading and not self.stop_worker.is_set():
                    self._predictive_load()

            except Exception as e:
                if self.debug:
                    print(f"BlockSwap async worker error: {e}")
                # Log the error and continue the loop, do not let the worker thread die
                time.sleep(1.0) # Wait a bit before retrying to avoid busy-loop on persistent error

    def _predictive_load(self):
        """Predictively load blocks that might be needed soon."""
        current_time = time.time()
        recently_used_threshold = 2.0  # seconds

        for name, block in self.blocks.items():
            # Check if it was recently used, is on CPU, not currently active, and not already in load queue
            if (current_time - block.last_used < recently_used_threshold and
                self._get_module_device_type(block.module) == 'cpu' and
                not block.is_active and
                name not in self.load_queue): # Prevent duplicate additions
                
                if self.debug:
                    print(f"BlockSwap: Adding '{name}' to load queue for predictive loading.")
                self.load_queue.append(name)

    @contextmanager
    def preserve_block(self, block_name: str):
        """Context manager to ensure a block stays on GPU."""
        block_info = self.blocks.get(block_name)
        if not block_info:
            yield
            return

        # Add to _preserved_blocks set
        self._preserved_blocks.add(block_name)

        # Ensure it's on GPU (if not already)
        if self._get_module_device_type(block_info.module) == 'cpu':
            self._swap_to_gpu(block_name, urgent=True)

        try:
            yield
        finally:
            # Remove from _preserved_blocks
            self._preserved_blocks.discard(block_name)
            # You might want to trigger a memory check here if releasing a large block
            if self.is_enabled:
                self._check_memory_pressure()


    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


def create_blockswap_manager(
    model: nn.Module,
    memory_threshold: float = 0.85,
    max_blocks_on_gpu: Optional[int] = None,
    use_pinned_memory: bool = True,
    **kwargs
) -> BlockSwapManager:
    """Create and configure a BlockSwapManager for the given model."""
    return BlockSwapManager(
        model=model,
        memory_threshold=memory_threshold,
        max_blocks_on_gpu=max_blocks_on_gpu,
        use_pinned_memory=use_pinned_memory,
        **kwargs
    )


# Convenience function for integration
def enable_blockswap(
    model: nn.Module,
    memory_threshold: float = 0.85,
    max_blocks_on_gpu: Optional[int] = None,
    use_pinned_memory: bool = True,
    **kwargs
) -> BlockSwapManager:
    manager = create_blockswap_manager(
        model, memory_threshold, max_blocks_on_gpu, use_pinned_memory=use_pinned_memory, **kwargs
    )
    manager.enable()
    return manager