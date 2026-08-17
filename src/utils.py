"""Utility functions for signature phrase discovery."""

import re
from typing import List


def hms_to_seconds(hms: str) -> float:
    """Convert a time string in HH:MM:SS format to seconds.
    
    Args:
        hms: Time string like "1:30:00" or "30:00" or "45"
        
    Returns:
        Time in seconds as a float
        
    Examples:
        hms_to_seconds("1:30:00") -> 5400.0
        hms_to_seconds("30:00") -> 1800.0
        hms_to_seconds("45") -> 45.0
    """
    if not hms:
        return 0.0
    
    # Split the time string by colons
    parts = [float(p) for p in hms.split(":")]
    
    # Multipliers for each position: seconds, minutes, hours
    multipliers = [1, 60, 3600]
    
    # Calculate total seconds
    # For example, [1, 30, 0] becomes 1*3600 + 30*60 + 0*1 = 5400
    return sum(p * multipliers[len(parts) - 1 - i] for i, p in enumerate(parts))


def extract_video_id(filename: str) -> str:
    """Extract YouTube video ID from a filename.
    
    YouTube video IDs are 11-character alphanumeric strings.
    They are often found in filenames like "video_[abc123xyz].txt"
    
    Args:
        filename: The filename to extract ID from
        
    Returns:
        The 11-character video ID, or the filename without extension if not found
    """
    # Try to find an 11-character ID in brackets like [abc123xyz]
    match = re.search(r"\[([A-Za-z0-9_-]{11})\]", filename)
    if match:
        return match.group(1)
    
    # If not found, return the filename without the .txt extension
    return filename.replace(".txt", "").split(".")[-1]


def tokenize(text: str) -> List[str]:
    """Convert text into a list of lowercase tokens (words).
    
    This function:
    1. Removes punctuation and special characters
    2. Converts everything to lowercase
    3. Splits into individual words
    
    Args:
        text: The text to tokenize
        
    Returns:
        List of lowercase word tokens
    """
    if not text:
        return []
    
    # Replace non-word characters (except apostrophes in words) with spaces
    # This keeps contractions like "don't" but removes other punctuation
    cleaned = re.sub(r"([^\w\s']|'(?!\w)|\s'+|'+\s)", " ", text)
    
    # Convert to lowercase and split into words
    return cleaned.lower().split()
