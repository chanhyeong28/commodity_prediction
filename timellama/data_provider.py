"""
Data provider module for TimeLlaMA.
This module provides the data_provider function expected by run.py.
"""

from timellama.data.dataloader import data_provider, mitsui_data_provider

# Use Mitsui data provider for the commodity prediction task
data_provider = mitsui_data_provider

# Export the data_provider function for backward compatibility
__all__ = ['data_provider']
