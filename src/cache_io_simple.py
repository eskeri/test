"""Compatibility alias: the simple cache lives in cache_io.

comparison_modes.py and ngram_counting.py import from cache_io_simple for
historical reasons; this just re-exports everything from cache_io so there is
one implementation.
"""

from .cache_io import *  # noqa: F401,F403
from .cache_io import (  # noqa: F401  (explicit re-export for star-ignore linters)
    get_cache_file_path,
    get_index_file_path,
    compute_data_hash,
    load_index,
    save_index,
    begin_cache_batch,
    flush_cache_batch,
    close_cache_connection,
    cache_exists,
    get_channel_info,
    write_channel_counts,
    write_channel_cache,
    read_channel_cache,
    read_channel_cache_filtered,
    get_all_channels,
    get_total_token_count,
    read_multiple_channels_filtered,
    read_aggregate_filtered,
)
