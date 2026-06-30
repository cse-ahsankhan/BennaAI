"""
Benna AI — Contract Risk & Clause Flagging Pipeline

Scans an uploaded contract against a curated GCC/FIDIC-aligned risk knowledge base.
Identifies missing clauses, unfair risk allocations, and high-risk provisions.

No external data source required — the knowledge base is compiled from publicly
available legal commentary on standard construction contract norms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import config
from ingest.embedder import embed_query
from retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk Knowledge Base
# Compiled from public legal commentary on FIDIC/GCC construction norms.
# Each entry defines a risk topic, search queries, severity if absent/flagged,
# and an LLM evaluation prompt.
# ---------------------------------------------------------------------------

RISK_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "LD_CAP",
        "topic": "Liquidated Damages Cap",
        "search_queries": [
            "liquidated damages cap limit maximum percentage",
            "delay damages cap ceiling",
            "LD cap contract value",
        ],
        "severity_if_missing": "HIGH",
        "description": "Standard GCC contracts typically cap LDs at 5–15% of contract value. Uncapped LDs expose the contractor to unlimited liability.",
        "eval_prompt": (
            "Evaluate the Liquidated Damages (LD) clause found in the contract.\n"
            "Check: (1) Is there a cap on LDs (typically 5–15% of contract value)? "
            "(2) Is the daily LD rate reasonable and proportionate? "
            "(3) Are there any provisions that make LDs the sole remedy for delay?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences describing what was found]\n"
            "RECOMMENDATION: [1 sentence on what the contracts team should do]"
        ),
    },
    {
        "id": "FORCE_MAJEURE",
        "topic": "Force Majeure",
        "search_queries": [
            "force majeure exceptional risk unforeseeable",
            "act of god natural disaster pandemic unforeseen",
            "relief event excusable delay",
        ],
        "severity_if_missing": "HIGH",
        "description": "Force Majeure clauses protect both parties from unforeseeable events (pandemics, war, extreme weather). Absence exposes the contractor to liability for events beyond their control.",
        "eval_prompt": (
            "Evaluate the Force Majeure clause in the contract.\n"
            "Check: (1) Is Force Majeure defined and does it include a broad enough scope of events? "
            "(2) Does it provide for extension of time AND/OR additional payment? "
            "(3) Is there a notice requirement and is it reasonable (e.g., 14–28 days)?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "DISPUTE_RESOLUTION",
        "topic": "Dispute Resolution Mechanism",
        "search_queries": [
            "dispute resolution arbitration adjudication DAB DAAB",
            "dispute board engineer determination",
            "claim arbitration DIAC ICC LCIA",
        ],
        "severity_if_missing": "HIGH",
        "description": "GCC contracts should specify a clear dispute resolution path: typically Engineer determination → DAB/DAAB → Arbitration. Absence forces parties into litigation.",
        "eval_prompt": (
            "Evaluate the Dispute Resolution mechanism in the contract.\n"
            "Check: (1) Is there a multi-tier dispute resolution process (e.g., Engineer → DAAB → Arbitration)? "
            "(2) Is the arbitration seat and rules clearly specified? "
            "(3) Are time limits for notices and decisions defined?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "DEFECTS_LIABILITY",
        "topic": "Defects Liability Period",
        "search_queries": [
            "defects liability period defects notification DNP",
            "maintenance period warranty defects",
            "rectification period after completion",
        ],
        "severity_if_missing": "MEDIUM",
        "description": "Standard GCC contracts specify a Defects Notification Period (DNP) of 12 months from Taking Over. Longer DNPs (24–36 months) shift risk heavily to the contractor.",
        "eval_prompt": (
            "Evaluate the Defects Liability Period (DLP/DNP) clause.\n"
            "Check: (1) What is the duration of the defects liability period? "
            "(2) Is it reasonable (standard is 12 months; anything over 24 months is unfair)? "
            "(3) Are there provisions for extension of the DLP?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "PAYMENT_TERMS",
        "topic": "Payment Terms & Timelines",
        "search_queries": [
            "payment certificate interim payment period days",
            "payment terms invoice due date interest overdue",
            "milestone payment progress payment schedule",
        ],
        "severity_if_missing": "HIGH",
        "description": "Standard GCC practice requires payment within 28–56 days of interim payment certificate. Extended payment periods or missing interest provisions create significant cash flow risk.",
        "eval_prompt": (
            "Evaluate the payment terms in the contract.\n"
            "Check: (1) What is the timeline for issuing and honouring payment certificates? "
            "(standard is 28 days to issue, 28 days to pay = 56 days total). "
            "(2) Is there a provision for interest on late payments? "
            "(3) Are there any conditions that allow the employer to withhold or set off payments without dispute resolution?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "TERMINATION_CONTRACTOR",
        "topic": "Termination for Convenience by Employer",
        "search_queries": [
            "termination convenience employer terminate contract",
            "termination for convenience payment on termination",
            "employer's right to terminate",
        ],
        "severity_if_missing": "MEDIUM",
        "description": "Termination for convenience clauses should specify compensation to the contractor (work done + loss of profit). Clauses that only pay for work done—or nothing—are high-risk.",
        "eval_prompt": (
            "Evaluate the Termination for Convenience clause.\n"
            "Check: (1) Does the Employer have a right to terminate for convenience? "
            "(2) What compensation does the Contractor receive on such termination? "
            "(standard = work done + materials + loss of anticipated profit on unexecuted work). "
            "(3) Are there any limitations on the contractor's ability to claim?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "EXTENSION_OF_TIME",
        "topic": "Extension of Time (EOT) Entitlement",
        "search_queries": [
            "extension of time EOT delay entitlement",
            "time extension contractor risk employer risk",
            "notice of delay clause time claim",
        ],
        "severity_if_missing": "HIGH",
        "description": "EOT clauses must clearly define which delay events give the contractor an entitlement to extra time (and cost). Narrow EOT provisions expose contractors to LDs for events they couldn't control.",
        "eval_prompt": (
            "Evaluate the Extension of Time (EOT) clause.\n"
            "Check: (1) What delay events entitle the Contractor to an EOT? "
            "(2) Are employer-caused delays, variation instructions, and unforeseeable conditions included? "
            "(3) Is the notice period for claiming EOT reasonable (standard is 28 days)? "
            "(4) Does EOT also entitle the contractor to additional cost?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "VARIATION_ORDERS",
        "topic": "Variation / Change Order Mechanism",
        "search_queries": [
            "variation order change instruction valuation",
            "instruction to vary scope change order price",
            "variation mechanism daywork rates",
        ],
        "severity_if_missing": "MEDIUM",
        "description": "A clear variation mechanism with defined valuation rules (BOQ rates, dayworks, or agreement) protects both parties. Absent or vague variation procedures lead to disputes.",
        "eval_prompt": (
            "Evaluate the Variation/Change Order clause.\n"
            "Check: (1) Can only the Engineer issue variation instructions? "
            "(2) Is there a defined hierarchy for valuing variations (e.g., BOQ rates → pro-rata → agreed rates → dayworks)? "
            "(3) Can the contractor object to a variation it considers impossible or unsafe? "
            "(4) Are there notice requirements for claiming additional cost arising from variations?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "LIMITATION_OF_LIABILITY",
        "topic": "Limitation of Liability",
        "search_queries": [
            "limitation of liability consequential loss indirect damages",
            "liability cap contractor employer",
            "exclude consequential special damages",
        ],
        "severity_if_missing": "HIGH",
        "description": "Contracts should contain mutual exclusions for indirect/consequential losses. Absence of a liability cap on the contractor's side creates open-ended financial exposure.",
        "eval_prompt": (
            "Evaluate the Limitation of Liability clause.\n"
            "Check: (1) Is there a mutual exclusion of consequential/indirect losses? "
            "(2) Is there an overall cap on the contractor's total liability under the contract? "
            "(standard is contract price, sometimes 150–200% for design-build). "
            "(3) Are there carve-outs from the cap (e.g., fraud, wilful misconduct, personal injury)?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "GOVERNING_LAW",
        "topic": "Governing Law & Jurisdiction",
        "search_queries": [
            "governing law jurisdiction applicable law",
            "law of UAE Saudi Arabia Qatar local law",
            "choice of law contract interpretation",
        ],
        "severity_if_missing": "MEDIUM",
        "description": "GCC projects should clearly state the governing law (e.g., UAE law, Saudi law). Ambiguity or references to foreign law create enforcement risk and legal uncertainty.",
        "eval_prompt": (
            "Evaluate the Governing Law and Jurisdiction clause.\n"
            "Check: (1) Is the governing law clearly specified? "
            "(2) Is it consistent with the project location (GCC jurisdiction)? "
            "(3) Is the dispute resolution jurisdiction (courts or arbitration seat) consistent with the governing law?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "PERFORMANCE_SECURITY",
        "topic": "Performance Security / Bond",
        "search_queries": [
            "performance bond security guarantee on-demand unconditional",
            "performance security bank guarantee amount percentage",
            "advance payment guarantee bond",
        ],
        "severity_if_missing": "LOW",
        "description": "Performance bonds are standard (typically 10% of contract value). On-demand (unconditional) bonds are high risk for the contractor; conditional bonds are preferred.",
        "eval_prompt": (
            "Evaluate the Performance Security (Bond) clause.\n"
            "Check: (1) What is the bond value (standard is 10% of contract price)? "
            "(2) Is the bond on-demand (unconditional) or conditional on proof of breach? "
            "(On-demand bonds are high risk for the contractor). "
            "(3) When is the bond released (typically at Taking Over or end of DNP)?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
    {
        "id": "INSURANCE",
        "topic": "Insurance Requirements",
        "search_queries": [
            "insurance contractor all risk works professional indemnity",
            "insurance requirements employer third party",
            "professional indemnity public liability insurance",
        ],
        "severity_if_missing": "MEDIUM",
        "description": "Standard GCC contracts require Contractor's All Risk (CAR), Third Party Liability, and sometimes Professional Indemnity. Missing or inadequate insurance requirements expose both parties.",
        "eval_prompt": (
            "Evaluate the Insurance clause.\n"
            "Check: (1) What insurance types are required (CAR, TPL, Professional Indemnity, Workers' Compensation)? "
            "(2) Are minimum coverage amounts specified? "
            "(3) Who is named as the insured and is the cross-liability clause present? "
            "(4) Are the insurance requirements commercially reasonable?\n"
            "Respond with:\n"
            "RISK_LEVEL: [HIGH / MEDIUM / LOW / OK]\n"
            "FINDING: [1-2 sentences]\n"
            "RECOMMENDATION: [1 sentence]"
        ),
    },
]


# ---------------------------------------------------------------------------
# Dataclass for a Risk Flag result
# ---------------------------------------------------------------------------

@dataclass
class RiskFlag:
    id: str
    topic: str
    severity: str          # HIGH / MEDIUM / LOW / OK / NOT_FOUND
    finding: str
    recommendation: str
    source_chunks: List[Dict[str, Any]] = field(default_factory=list)
    is_missing: bool = False


# ---------------------------------------------------------------------------
# LLM helper (reuse conflict pipeline's direct caller pattern)
# ---------------------------------------------------------------------------

def _call_llm_direct(provider: str, system: str, user: str) -> str:
    if provider == "claude":
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    elif provider == "ollama":
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError("langchain-community required for Ollama")
        llm = Ollama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            num_ctx=4096,
            num_predict=512,
        )
        prompt = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}\n\nResponse:"
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}."
                ) from exc
            raise

    elif provider == "mock":
        # Deterministic mock response for testing
        return (
            "RISK_LEVEL: MEDIUM\n"
            "FINDING: This is a simulated risk scan result for testing purposes. "
            "In a real scan, the LLM would evaluate the actual clause text.\n"
            "RECOMMENDATION: Review this clause with your contracts engineer before signing."
        )

    else:
        raise ValueError(f"Unknown provider: '{provider}'")


def _parse_risk_response(text: str) -> Dict[str, str]:
    """Parse structured LLM risk evaluation response."""
    import re
    result = {"risk_level": "UNCLEAR", "finding": "", "recommendation": ""}

    risk_match = re.search(
        r"RISK_LEVEL:\s*(HIGH|MEDIUM|LOW|OK|UNCLEAR|NOT_FOUND)",
        text, re.IGNORECASE
    )
    if risk_match:
        result["risk_level"] = risk_match.group(1).upper()

    finding_match = re.search(
        r"FINDING:\s*(.+?)(?=\nRECOMMENDATION:|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if finding_match:
        result["finding"] = finding_match.group(1).strip()

    rec_match = re.search(
        r"RECOMMENDATION:\s*(.+?)$",
        text, re.IGNORECASE | re.DOTALL
    )
    if rec_match:
        result["recommendation"] = rec_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_contract_for_risks(
    project_id: str,
    source_file: str,
    provider: Optional[str] = None,
    top_k: int = 3,
) -> List[RiskFlag]:
    """
    Scan a contract document for risk flags across all topics in the knowledge base.

    Args:
        project_id:  Benna AI project namespace.
        source_file: The document to scan (must be indexed in the project).
        provider:    LLM provider ('claude' | 'ollama' | 'mock').
        top_k:       Number of chunks to retrieve per risk topic.

    Returns:
        List of RiskFlag objects, sorted HIGH → MEDIUM → LOW → OK.
    """
    provider = (provider or config.LLM_PROVIDER).lower()
    doc_filter = {"source_file": source_file}
    flags: List[RiskFlag] = []

    _RISK_SYSTEM = (
        "You are a GCC construction contracts risk analyst. "
        "You are evaluating specific clauses from a construction contract "
        "against standard FIDIC/GCC practice norms. "
        "Be concise, specific, and practical. "
        "Always cite specific clause language you observed in the excerpts."
    )

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3, "NOT_FOUND": 1, "UNCLEAR": 2}

    for topic_def in RISK_KNOWLEDGE_BASE:
        topic_id = topic_def["id"]
        topic_name = topic_def["topic"]
        logger.info("Risk scan [%s]: evaluating topic '%s'", source_file, topic_name)

        # 1. Retrieve relevant chunks using the best search query
        best_chunks = []
        for search_query in topic_def["search_queries"]:
            try:
                embedding = embed_query(search_query)
                chunks = hybrid_search(
                    query=search_query,
                    query_embedding=embedding,
                    project_id=project_id,
                    top_k_each=top_k,
                    final_top_k=top_k,
                    filters=doc_filter,
                )
                if chunks and (not best_chunks or chunks[0].get("rrf_score", 0) > best_chunks[0].get("rrf_score", 0)):
                    best_chunks = chunks
            except Exception as exc:
                logger.warning("Retrieval error for topic '%s': %s", topic_name, exc)

        # 2. If no relevant content found, flag as missing
        if not best_chunks:
            flags.append(RiskFlag(
                id=topic_id,
                topic=topic_name,
                severity=topic_def["severity_if_missing"],
                finding=f"No clause relating to '{topic_name}' was found in the document. This is a required standard provision.",
                recommendation=f"Add a '{topic_name}' clause before contract execution. Refer to FIDIC Sub-Clause standards for GCC projects.",
                source_chunks=[],
                is_missing=True,
            ))
            continue

        # 3. Format the retrieved chunks into context
        context_parts = []
        for i, chunk in enumerate(best_chunks, 1):
            meta = chunk.get("metadata", {})
            clause = meta.get("clause_ref", "")
            clause_str = f" | Clause {clause}" if clause else ""
            context_parts.append(
                f"[{i}] Page {meta.get('page_num', '?')}{clause_str}\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # 4. Build and send the evaluation prompt
        user_prompt = (
            f"Contract Excerpts relating to '{topic_name}':\n\n"
            f"{context}\n\n"
            f"--- EVALUATION TASK ---\n"
            f"{topic_def['eval_prompt']}\n\n"
            f"Standard benchmark: {topic_def['description']}"
        )

        try:
            response_text = _call_llm_direct(provider, _RISK_SYSTEM, user_prompt)
            parsed = _parse_risk_response(response_text)
        except Exception as exc:
            logger.warning("LLM evaluation error for topic '%s': %s", topic_name, exc)
            parsed = {
                "risk_level": "UNCLEAR",
                "finding": f"Could not evaluate this clause due to an error: {exc}",
                "recommendation": "Review manually.",
            }

        flags.append(RiskFlag(
            id=topic_id,
            topic=topic_name,
            severity=parsed["risk_level"],
            finding=parsed["finding"],
            recommendation=parsed["recommendation"],
            source_chunks=best_chunks,
            is_missing=False,
        ))

    # Sort by severity
    flags.sort(key=lambda f: severity_order.get(f.severity, 99))
    return flags
