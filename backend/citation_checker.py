"""
Citation Checker Module
Extracts and verifies academic citations from text.
Checks: author names, year, title, venue, page numbers
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Tuple
from collections import Counter
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

CITATION_EXTRACT_PROMPT = """You are an expert at parsing academic citations. Extract citation details from the given text.

TEXT:
{text}

Extract ALL citations found and return a JSON array. For each citation, extract:
- raw_citation: The exact citation text as it appears
- authors: Full author list as written (e.g., "He, K., Zhang, X., Ren, S., & Sun, J.")
- year: Publication year (4 digits as string)
- title: Paper/article title
- venue: Journal name, conference name, or publisher (e.g., "CVPR", "Nature", "IEEE")
- pages: Page range if present (e.g., "770-778")

Return ONLY a valid JSON array like this:
[
  {{
    "raw_citation": "He, K., et al. (2016). Deep residual learning...",
    "authors": "He, K., Zhang, X., Ren, S., & Sun, J.",
    "year": "2016",
    "title": "Deep residual learning for image recognition",
    "venue": "CVPR",
    "pages": "770-778"
  }}
]

If no citations found, return: []
Return ONLY valid JSON, no other text."""

CITATION_VERIFY_PROMPT = """You are an expert academic citation verifier. Check if the citation details are accurate.

CITATION TO VERIFY:
- Authors: {authors}
- Year: {year}
- Title: {title}
- Venue: {venue}
- Pages: {pages}

SEARCH RESULTS FROM THE WEB:
{search_results}

VERIFICATION TASK:
Compare the citation details against the search results. Check for:
1. Is the YEAR correct? (Common error: off by 1 year)
2. Is the VENUE correct? (Common error: wrong conference/journal)
3. Are the PAGE NUMBERS correct? (Common error: off by 1 page)
4. Are the AUTHORS correct?
5. Is the TITLE accurate?

RESPOND with ONLY a JSON object:
{{
  "status": "VERIFIED|HALLUCINATED|UNVERIFIABLE",
  "errors": ["list of specific errors found, if any"],
  "reason": "Brief explanation under 150 characters"
}}

STATUS RULES:
- VERIFIED: All citation details match search results
- HALLUCINATED: One or more details are WRONG (incorrect year, venue, pages, etc.)
- UNVERIFIABLE: Cannot find enough evidence to verify

Return ONLY valid JSON."""


async def extract_citations(text: str) -> List[Dict]:
    """
    Extract academic citations from text using LLM.
    
    Args:
        text: Text containing citations
        
    Returns:
        List of citation dicts with author, year, title, venue, pages
    """
    if not text.strip():
        return []
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": CITATION_EXTRACT_PROMPT.format(text=text)
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            citations = json.loads(json_match.group())
            return citations
        
        return json.loads(content)
        
    except Exception as e:
        print(f"Error extracting citations: {e}")
        return []


async def verify_citation_with_model(citation: Dict, search_results: str, temperature: float) -> Tuple[str, List[str], str]:
    """
    Verify a citation with a specific temperature setting.
    
    Returns:
        Tuple of (status, errors list, reason)
    """
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": CITATION_VERIFY_PROMPT.format(
                        authors=citation.get("authors", "Unknown"),
                        year=citation.get("year", "Unknown"),
                        title=citation.get("title", "Unknown"),
                        venue=citation.get("venue", "Unknown"),
                        pages=citation.get("pages", "Unknown"),
                        search_results=search_results
                    )
                }
            ],
            temperature=temperature,
            max_tokens=512
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(content)
        
        status = result.get("status", "UNVERIFIABLE").upper()
        if status not in ["VERIFIED", "HALLUCINATED", "UNVERIFIABLE"]:
            status = "UNVERIFIABLE"
        
        errors = result.get("errors", [])
        reason = result.get("reason", "Unable to verify")[:150]
        
        return (status, errors, reason)
        
    except Exception as e:
        print(f"Error verifying citation (temp={temperature}): {e}")
        return ("UNVERIFIABLE", [], f"Error: {str(e)[:100]}")


async def verify_citation(citation: Dict, search_results: List[Dict]) -> Dict:
    """
    Verify a single citation using multi-model voting.
    
    Args:
        citation: Citation dict with author, year, title, venue, pages
        search_results: List of search results
        
    Returns:
        Dict with status, errors, reason
    """
    if not search_results:
        return {
            "status": "UNVERIFIABLE",
            "errors": [],
            "reason": "No search results found to verify this citation"
        }
    
    # Format search results
    formatted_results = "\n".join([
        f"- {r['title']}: {r['snippet']}"
        for r in search_results
    ])
    
    # Run 3 verification calls with different temperatures
    temperatures = [0.1, 0.3, 0.5]
    
    try:
        tasks = [
            verify_citation_with_model(citation, formatted_results, temp)
            for temp in temperatures
        ]
        results = await asyncio.gather(*tasks)
        
        # Extract statuses
        statuses = [r[0] for r in results]
        all_errors = []
        for r in results:
            all_errors.extend(r[1])
        reasons = {r[0]: r[2] for r in results}
        
        # Vote on status
        vote_counts = Counter(statuses)
        majority_status = vote_counts.most_common(1)[0][0]
        vote_count = vote_counts[majority_status]
        
        # Deduplicate errors
        unique_errors = list(set(all_errors))
        
        # Create reason
        if vote_count == 3:
            reason = f"All 3 checks agree: {reasons[majority_status]}"
        elif vote_count == 2:
            reason = f"2/3 checks agree: {reasons[majority_status]}"
        else:
            reason = f"Checks disagree. {majority_status}: {reasons[majority_status]}"
        
        return {
            "status": majority_status,
            "errors": unique_errors,
            "reason": reason[:150]
        }
        
    except Exception as e:
        print(f"Error in citation verification: {e}")
        return {
            "status": "UNVERIFIABLE",
            "errors": [],
            "reason": f"Error: {str(e)[:100]}"
        }
