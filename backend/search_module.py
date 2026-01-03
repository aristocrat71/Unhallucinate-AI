"""
Search Module
INPUT: String (a claim)
OUTPUT: List[Dict] with title, url, snippet
CONSTRAINT: Top 3 results, <5 second timeout
"""

import asyncio
from typing import List, Dict
from duckduckgo_search import DDGS


async def search_web(claim: str, max_results: int = 3, timeout: int = 5) -> List[Dict]:
    """
    Search the web for evidence about a claim using DuckDuckGo.
    
    Args:
        claim: The claim to search for
        max_results: Maximum number of results (default 3)
        timeout: Search timeout in seconds (default 5)
        
    Returns:
        List of dicts with title, url, snippet
    """
    if not claim.strip():
        return []
    
    try:
        # Run synchronous DuckDuckGo search in executor
        loop = asyncio.get_event_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_search, claim, max_results),
            timeout=timeout
        )
        return results
        
    except asyncio.TimeoutError:
        print(f"Search timeout for: {claim[:50]}...")
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []


def _sync_search(query: str, max_results: int) -> List[Dict]:
    """
    Synchronous DuckDuckGo search (runs in executor).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
    except Exception as e:
        print(f"DuckDuckGo error: {e}")
        return []
