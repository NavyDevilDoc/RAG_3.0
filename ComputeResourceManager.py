# ComputeResourceManager.py

"""
ComputeResourceManager.py

A module for managing compute resources and GPU configuration in deep learning applications.

Features:
- GPU availability detection
- Memory usage monitoring
- Resource optimization
- CUDA configuration management
- Automatic device selection
"""

import gc
import torch
import logging
from typing import Dict

class ComputeResourceManager:
    """
    Manages compute resources and GPU configuration for deep learning tasks.

    Features:
    1. GPU detection and configuration
    2. Memory usage monitoring
    3. Resource optimization
    4. Device selection
    5. CUDA settings management

    Attributes:
        logger (logging.Logger): Logger instance
        has_gpu (bool): Whether GPU is available
        device (torch.device): Selected compute device

    Example:
        >>> manager = ComputeResourceManager()
        >>> manager.test_gpu_details()
        >>> settings = manager.get_compute_settings()
    """
    
    def __init__(self):
        """
        Initialize compute resource manager with logging and device detection.

        Raises:
            RuntimeError: If CUDA initialization fails
        """
        self.logger = logging.getLogger(__name__)
        self.has_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.has_gpu else "cpu")


    def test_gpu_details(self) -> None:
        """
        Test and display GPU/CUDA configuration and memory details.

        Outputs:
            - CUDA version and availability
            - GPU device information
            - Memory usage statistics
            - Cache status

        Raises:
            RuntimeError: If GPU query fails

        Example:
            >>> manager = ComputeResourceManager()
            >>> manager.test_gpu_details()
        """
        print("\n=== CUDA Configuration ===")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA available: {self.has_gpu}")
        
        if self.has_gpu:
            # Display GPU device information
            print("\n=== GPU Details ===")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Torch version: {torch.__version__}")
            
            # Display memory usage statistics
            print("\n=== GPU Memory Usage ===")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
            print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Clean up GPU memory
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("\nNo GPU available - running on CPU only")


    def get_compute_settings(self) -> Dict:
        """
        Detect available compute resources and return optimal settings.

        Returns:
            Dict: Configuration settings including:
                - temperature (float): Sampling temperature
                - top_k (int): Top K sampling parameter
                - top_p (float): Nucleus sampling parameter
                - num_gpu (int): Number of GPUs to use
                - num_thread (int): Number of CPU threads
                - max_tokens (int): Maximum token limit

        Raises:
            RuntimeError: If resource detection fails

        Example:
            >>> manager = ComputeResourceManager()
            >>> settings = manager.get_compute_settings()
            >>> print(f"Using {settings['num_gpu']} GPUs")
        """
        # Set default parameters for sampling behavior
        settings = {
            'temperature': 0.1,
            'top_k': 10,
            'top_p': 0.2
        }
        
        try:
            if self.has_gpu:
                # If GPU available, print device name and apply GPU-optimized settings
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU detected: {gpu_name}. Using GPU acceleration.")
                settings.update({
                    'num_gpu': 1,
                    'num_thread': 8,  # Reduced threads for GPU-primary processing
                    'max_tokens': 2048  # Token limit for efficient GPU processing
                })
            else:
                # No GPU, revert to CPU-optimized configuration
                print("No GPU detected. Using CPU only.")
                settings.update({
                    'num_gpu': 0,
                    'num_thread': 16  # Increased threads for CPU-only environment
                })
            return settings
        except Exception as e:
            # If detection fails, log error and default to CPU settings
            self.logger.error(f"Error detecting compute resources: {e}")
            settings.update({
                'num_gpu': 0,
                'num_thread': 16
            })
            return settings

    def clear_gpu_memory(self) -> None:
        """
        Clear GPU memory cache if GPU is available.

        Performs garbage collection and CUDA cache clearing to free up GPU memory.

        Returns:
            None

        Example:
            >>> manager = ComputeResourceManager()
            >>> # After heavy GPU operations
            >>> manager.clear_gpu_memory()
    """
        if self.has_gpu:
            gc.collect()
            torch.cuda.empty_cache()
