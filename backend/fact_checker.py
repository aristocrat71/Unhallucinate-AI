"""
Fact Checker Module
INPUT: Claim string + search results
OUTPUT: {status, reason}
CONSTRAINT: status must be exactly one of: VERIFIED | HALLUCINATED | UNVERIFIABLE
"""

import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

FACT_CHECK_PROMPT = """You are a rigorous fact-checking assistant. Analyze whether the ENTIRE claim is supported by search results.

CLAIM TO VERIFY:
{claim}

SEARCH RESULTS:
{search_results}

VERIFICATION RULES:
1. VERIFIED - The search results CLEARLY and DIRECTLY support the COMPLETE claim
   - All parts of the claim must be confirmed (subject, action, object, time, place, etc.)
   - Example: "Einstein discovered penicillin" requires proof Einstein discovered it (not just that Einstein existed)

2. HALLUCINATED - The search results CONTRADICT the claim OR show it's factually wrong
   - Any part of the claim that is proven false makes the whole claim HALLUCINATED
   - Example: If claim says "X did Y" but sources say "Z did Y", mark as HALLUCINATED

3. UNVERIFIABLE - Not enough evidence in search results to confirm or deny
   - Sources don't mention the specific claim at all
   - Sources are ambiguous or inconclusive

CRITICAL: Verify the COMPLETE STATEMENT, not just that entities exist!
- "Einstein discovered penicillin" is HALLUCINATED even though Einstein existed
- "Musk founded Google" is HALLUCINATED even though both Musk and Google exist

Respond with ONLY a JSON object in this exact format:
{{"status": "VERIFIED|HALLUCINATED|UNVERIFIABLE", "reason": "Brief explanation under 150 characters"}}

IMPORTANT:
- status MUST be exactly one of: VERIFIED, HALLUCINATED, UNVERIFIABLE
- reason MUST be under 150 characters explaining why
- Return ONLY valid JSON, no other text"""


async def check_fact(claim: str, search_results: List[Dict]) -> Dict:
    """
    Check a claim against search results using Groq LLM.
    
    Args:
        claim: The claim to verify
        search_results: List of search results with title, url, snippet
        
    Returns:
        Dict with status and reason
    """
    # If no search results, mark as unverifiable
    if not search_results:
        return {
            "status": "UNVERIFIABLE",
            "reason": "No search results found to verify this claim"
        }
    
    # Format search results for the prompt
    formatted_results = "\n".join([
        f"- {r['title']}: {r['snippet']}"
        for r in search_results
    ])
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": FACT_CHECK_PROMPT.format(
                        claim=claim,
                        search_results=formatted_results
                    )
                }
            ],
            temperature=0.1,
            max_tokens=256
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(content)
        
        # Validate status
        valid_statuses = ["VERIFIED", "HALLUCINATED", "UNVERIFIABLE"]
        status = result.get("status", "UNVERIFIABLE").upper()
        if status not in valid_statuses:
            status = "UNVERIFIABLE"
        
        # Truncate reason if too long
        reason = result.get("reason", "Unable to determine verification status")[:150]
        
        return {
            "status": status,
            "reason": reason
        }
        
    except json.JSONDecodeError:
        return {
            "status": "UNVERIFIABLE",
            "reason": "Error parsing fact-check response"
        }
    except Exception as e:
        print(f"Error checking fact: {e}")
        return {
            "status": "UNVERIFIABLE",
            "reason": f"Error during verification: {str(e)[:100]}"
        }
