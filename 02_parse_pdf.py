import re

sample_text = """
1. Introduction
This system uses a distributed cache to reduce latency across regions.

2. Configuration
The cache TTL is set via the CACHE_TTL environment variable, default 3600 seconds.

Table 1: Supported Regions
Region      | Latency (ms) | Status
us-east-1   | 12           | Active
eu-west-1   | 45           | Active
ap-south-1  | 78           | Beta

3. Error Handling
If a cache miss occurs, the system falls back to the primary database with a circuit breaker pattern.
"""

def structure_aware_chunk(text):
    # split on numbered headings like "1. Introduction", "2. Configuration"
    heading_pattern = r'\n(?=\d+\.\s)'
    raw_sections = re.split(heading_pattern, text.strip())

    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # if this section contains a table, keep it as one atomic chunk
        # (naive detection: presence of a line with multiple "|" separators)
        chunks.append(section)

    return chunks

chunks = structure_aware_chunk(sample_text)

for i, c in enumerate(chunks):
    print(f"---- Chunk {i+1} ----")
    print(c)
    print()