"""
Search Module - Web search for fact verification
INPUT: String (a claim)
OUTPUT: List[Dict] with title, url, snippet
CONSTRAINT: Top 3 results, <5 second timeout
Uses SerpAPI (Google) with DuckDuckGo as fallback
"""

import os
import asyncio
import aiohttp
from typing import List, Dict
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import ssl
import certifi

load_dotenv()

# SerpAPI for Google Search (free tier: 100 queries/month)
SERP_API_KEY = os.getenv("SERP_API_KEY", "")


async def search_web(query: str, max_results: int = 3, timeout: int = 5) -> List[Dict]:
    """
    Search the web for information about a claim.
    Uses SerpAPI (Google) if available, falls back to DuckDuckGo.
    
    Args:
        query: The search query (claim to verify)
        max_results: Number of results to return (default 3)
        timeout: Timeout in seconds
        
    Returns:
        List of dicts with title, url, snippet
    """
    if not query.strip():
        return []
    
    # Try SerpAPI (Google) first for better English results
    if SERP_API_KEY:
        results = await search_serpapi(query, max_results, timeout)
        if results:
            return results
    
    # Fallback to DuckDuckGo with English region
    return await search_duckduckgo(query, max_results, timeout)


async def search_serpapi(query: str, max_results: int = 3, timeout: int = 5) -> List[Dict]:
    """Search using SerpAPI (Google Search)."""
    try:
        url = "https://serpapi.com/search"
        params = {
            "api_key": SERP_API_KEY,
            "q": query,
            "engine": "google",
            "num": max_results,
            "hl": "en",  # English results
            "gl": "us",  # US region
        }
        
        # Create SSL context to fix Windows DNS/SSL issues with aiohttp
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get("organic_results", [])[:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                    print(f"  SerpAPI returned {len(results)} results")
                    return results
                else:
                    error_text = await response.text()
                    print(f"  SerpAPI error: HTTP {response.status}")
                    print(f"  SerpAPI response: {error_text[:200]}")
    except Exception as e:
        print(f"  SerpAPI search error: {type(e).__name__}: {e}")
    return []


async def search_duckduckgo(query: str, max_results: int = 3, timeout: int = 5) -> List[Dict]:
    """Search using DuckDuckGo with English region preference."""
    try:
        loop = asyncio.get_event_loop()
        
        def do_search():
            with DDGS() as ddgs:
                # Force English results with region parameter
                results = list(ddgs.text(
                    query, 
                    region='wt-wt',  # Worldwide English
                    safesearch='moderate',
                    max_results=max_results
                ))
                return results
        
        raw_results = await asyncio.wait_for(
            loop.run_in_executor(None, do_search),
            timeout=timeout
        )
        
        results = []
        for r in raw_results[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })
        
        print(f"  DuckDuckGo returned {len(results)} results")
        return results
        
    except asyncio.TimeoutError:
        print(f"  DuckDuckGo timeout for: {query[:50]}...")
        return []
    except Exception as e:
        print(f"  DuckDuckGo search error: {e}")
        return []


async def search_for_citation(citation_text: str, max_results: int = 5) -> List[Dict]:
    """
    Specialized search for academic citations.
    Searches academic sources to verify citation details.
    
    Args:
        citation_text: The full citation text
        max_results: Number of results to return
        
    Returns:
        List of search results from academic sources
    """
    # Extract key parts for targeted search
    queries = [
        f'"{citation_text[:100]}"',  # Exact match on first part
        f'{citation_text} site:scholar.google.com',  # Scholar
        f'{citation_text} site:semanticscholar.org',  # Semantic Scholar
    ]
    
    all_results = []
    for query in queries[:2]:  # Limit queries
        results = await search_web(query, max_results=3)
        all_results.extend(results)
        await asyncio.sleep(0.2)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)
    
    return unique_results[:max_results]
