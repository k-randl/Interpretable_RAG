import csv
from typing import Optional, Tuple

#====================================================================================================#
# File utilities:                                                                                    #
#====================================================================================================#

def sniff_file_dialect(filepath:str, sample_size:int=2048) -> Tuple[Optional[str], Optional[bool]]:
    """Sniffs a file to detect its delimiter and whether it has a header.

    Args:
        filepath (str):     The path to the file.
        sample_size (int):  How many bytes to read for the sample (default: `2048`).

    Returns:
        A tuple `(delimiter, has_header)`, or `(None, None)` on failure.
        - `delimiter`:  The detected delimiter (e.g. `','`, `'\\t'`), or `None` if it could not be detected.
        - `has_header`: `True` if a header is likely, `False` if not, `None` if uncertain or on error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(sample_size)

            sniffer    = csv.Sniffer()
            dialect    = sniffer.sniff(sample)
            has_header = sniffer.has_header(sample)

            return dialect.delimiter, has_header

    except csv.Error:
        print(f'Error: Could not detect a valid dialect for file: {filepath}')
        return None, None

    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None