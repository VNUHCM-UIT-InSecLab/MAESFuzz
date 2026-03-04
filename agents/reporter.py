#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return slug or "contract"



ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
from provider import context as provider_context  # noqa: E402
from provider import factory as provider_factory  # noqa: E402
from provider.base import ProviderBase, ProviderError, ProviderResult  # noqa: E402


LOGGER = logging.getLogger("ReporterAgent")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_reporter_system_prompt(threshold_high: float = 70.0, threshold_low: float = 30.0) -> str:
    """
    Build system prompt for LLM-as-a-Judge with confidence scoring.
    
    Args:
        threshold_high: Confidence threshold for CONFIRMED verdict (0-100)
        threshold_low: Confidence threshold for REJECTED verdict (0-100)
    
    Returns:
        System prompt string
    """
    return (
        "You are the UniFuzz Security Reporter Agent acting as an LLM-as-a-Judge.\n"
        "\n"
        "You receive ONE JSON payload describing ONE smart contract fuzzing session.\n"
        "The payload includes:\n"
        "- Solidity source code (possibly truncated)\n"
        "- Fuzzing metrics (coverage, time, generations)\n"
        "- A list of detected bugs with:\n"
        "  - SWC ID and vulnerability type\n"
        "  - Triggering transaction sequence (execution trace)\n"
        "  - Source code locations and snippets\n"
        "\n"
        "Your task is to evaluate each reported bug and produce an evidence-grounded,\n"
        "verifiable security report.\n"
        "\n"
        "────────────────────────────────\n"
        "CORE PRINCIPLES (MANDATORY)\n"
        "────────────────────────────────\n"
        "\n"
        "1. Evidence-Grounded Reasoning\n"
        "- Base ALL judgments and explanations strictly on the provided JSON payload.\n"
        "- Do NOT assume facts, behaviors, or vulnerabilities not explicitly supported\n"
        "  by execution traces or source code.\n"
        "- Every explanation MUST reference concrete execution evidence\n"
        "  (transaction order, call behavior, state changes, or trace symptoms).\n"
        "\n"
        "2. Faithfulness Requirement (Critical)\n"
        "- Your explanation must be semantically aligned with the execution trace.\n"
        "- Describe WHAT happened during execution, in WHAT ORDER, and WHY this leads\n"
        "  to the reported vulnerability.\n"
        "- Do NOT introduce speculative reasoning, hypothetical attacks, or abstract\n"
        "  vulnerability descriptions detached from the trace.\n"
        "\n"
        "3. Label Preservation\n"
        "- Do NOT change the SWC ID or vulnerability type.\n"
        "- Your role is to judge confidence and explainability, not reclassify bugs.\n"
        "\n"
        "4. Conservative Judgment\n"
        "- If evidence is weak, incomplete, or ambiguous, LOWER confidence and prefer\n"
        "  INCONCLUSIVE over CONFIRMED.\n"
        "- If key execution steps are missing, explicitly state this in the explanation.\n"
        "\n"
        "────────────────────────────────\n"
        "VERDICT POLICY\n"
        "────────────────────────────────\n"
        "\n"
        "- Output a confidence score between 0 and 100.\n"
        f"- CONFIRMED   if confidence >= {threshold_high}.\n"
        f"- REJECTED    if confidence < {threshold_low}.\n"
        "- INCONCLUSIVE otherwise\n"
        "\n"
        "Use the provided thresholds:\n"
        f"- τ_high = {threshold_high}\n"
        f"- τ_low  = {threshold_low}\n"
        "\n"
        "────────────────────────────────\n"
        "OUTPUT FORMAT (STRICT)\n"
        "────────────────────────────────\n"
        "\n"
        "Write the report in Markdown using EXACTLY the following structure and order:\n"
        "\n"
        "## Summary\n"
        "- Total bugs evaluated: X\n"
        "- CONFIRMED: Y | INCONCLUSIVE: Z | REJECTED: W\n"
        "- Overall risk level: LOW / MEDIUM / HIGH\n"
        "- Coverage summary (if available)\n"
        "\n"
        "## Bug Judgement\n"
        "\n"
        "For EACH bug, output ONE block using this template:\n"
        "\n"
        "- [VERDICT] SWC-{id}: {vulnerability_type}. Conf {score}/100. Sev {severity}.\n"
        "  Loc {function}:{line_range}.\n"
        "\n"
        "  **Evidence:**\n"
        "  - Describe the triggering transaction sequence step-by-step.\n"
        "  - Explicitly reference:\n"
        "    • caller identity (EOA / contract)\n"
        "    • function invocation order\n"
        "    • external calls or state updates\n"
        "    • trace symptoms (e.g., re-entry before state update)\n"
        "  - Use ONLY information present in the execution trace or source code.\n"
        "\n"
        "  **Reasoning:**\n"
        "  - Explain WHY the observed execution behavior constitutes this vulnerability.\n"
        "  - Connect execution order and state changes to the vulnerability definition.\n"
        "  - Avoid generic descriptions; stay trace-specific.\n"
        "\n"
        "  **Fix:**\n"
        "  - Provide ONE concrete and actionable remediation step\n"
        "    (e.g., reorder state update, add check, restrict caller).\n"
        "\n"
        "## Recommendations\n"
        "- Group findings by theme if possible.\n"
        "- Focus on concrete, implementable security improvements.\n"
        "\n"
        "## RAG Observations\n"
        "- Briefly describe what security patterns or stateful behaviors\n"
        "  the generated tests appear to target.\n"
        "- Do NOT mention prompts, model internals, or hallucinations.\n"
        "\n"
        "────────────────────────────────\n"
        "STYLE RULES\n"
        "────────────────────────────────\n"
        "\n"
        "- Use bullet points starting with \"- \".\n"
        "- Be concise but precise.\n"
        "- Do NOT cite external sources.\n"
        "- Do NOT mention that you are an AI or language model.\n"
        "- Do NOT fabricate execution details.\n"
        "\n"
        "Your goal is to produce explanations that are:\n"
        "- Interpretable by humans\n"
        "- Directly verifiable against execution traces\n"
        "- Suitable for automatic faithfulness evaluation\n"
    )


# Default system prompt (will be replaced by build_reporter_system_prompt with default thresholds)
REPORTER_SYSTEM_PROMPT = build_reporter_system_prompt()


@dataclass
class SessionSummary:
    contract_label: str
    contract_path: str
    result_path: str
    result_json: Dict[str, Any]
    result_text: str
    contract_source: Optional[str]
    analysis_path: Optional[str]
    analysis_text: Optional[str]
    log_path: Optional[str]
    log_text: Optional[str]
    rag_storage_path: Optional[str]
    duplication_mode: Optional[str]
    fuzz_time: Optional[int]
    max_trans_length: Optional[int]
    provider_model: Optional[str] = None
    provider_name: Optional[str] = None
    reporter_enabled: bool = False
    technical_notes: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: _utc_now().strftime("%Y%m%d%H%M%S"))
    timestamp: datetime = field(default_factory=_utc_now)
    threshold_high: float = 70.0
    threshold_low: float = 30.0


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_contract_section(result_payload: Dict[str, Any], contract_label: str) -> Tuple[str, Dict[str, Any]]:
    """
    Locate the contract-specific section inside a UniFuzz result.json payload.

    - If there is an exact key match, use it.
    - If there is only one top-level entry, use that.
    """
    if contract_label in result_payload:
        return contract_label, result_payload[contract_label]
    if len(result_payload) == 1:
        key = next(iter(result_payload))
        return key, result_payload[key]
    raise KeyError(f"Cannot locate contract section for '{contract_label}'")


def _extract_metrics_from_contract_section(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw contract section from result.json into a compact metrics dict
    suitable for LLM consumption.
    """
    metrics: Dict[str, Any] = {}

    tx_info = section.get("transactions") or {}
    metrics["transactions_total"] = tx_info.get("total")
    metrics["transactions_per_second"] = tx_info.get("per_second")

    generations = section.get("generations") or []
    metrics["generations_count"] = len(generations)

    code_cov = section.get("code_coverage") or {}
    metrics["code_coverage_percentage"] = code_cov.get("percentage")
    metrics["code_coverage_covered"] = code_cov.get("covered")
    metrics["code_coverage_total"] = code_cov.get("total")

    branch_cov = section.get("branch_coverage") or {}
    metrics["branch_coverage_percentage"] = branch_cov.get("percentage")
    metrics["branch_coverage_covered"] = branch_cov.get("covered")
    metrics["branch_coverage_total"] = branch_cov.get("total")

    metrics["execution_time"] = section.get("execution_time")
    metrics["memory_consumption_mb"] = section.get("memory_consumption")

    # time_to_first_bug: min over all errors if available
    first_bug_time: Optional[float] = None
    errors = section.get("errors") or {}
    for _loc, entries in errors.items():
        for entry in entries:
            t = entry.get("time")
            if isinstance(t, (int, float)):
                if first_bug_time is None or t < first_bug_time:
                    first_bug_time = t
    metrics["time_to_first_bug"] = first_bug_time

    return metrics


def _extract_bugs_from_contract_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert the 'errors' structure of a UniFuzz contract result into a flat list
    of bugs that an LLM can judge.
    """
    bugs: List[Dict[str, Any]] = []
    errors = section.get("errors") or {}
    for loc, entries in errors.items():
        for idx, entry in enumerate(entries):
            bug: Dict[str, Any] = {}
            bug_id = f"{loc}_{idx}"
            bug["id"] = bug_id
            bug["location"] = loc
            bug["swc_id"] = entry.get("swc_id")
            bug["type"] = entry.get("type")
            bug["severity"] = entry.get("severity")
            bug["time"] = entry.get("time")
            bug["line"] = entry.get("line")
            bug["column"] = entry.get("column")
            bug["source_code"] = entry.get("source_code")

            # Individual transaction sequence can be very large; keep a short excerpt.
            individuals = entry.get("individual") or []
            bug["trigger_transactions_count"] = len(individuals)
            if individuals:
                bug["trigger_example"] = individuals[:2]
            else:
                bug["trigger_example"] = []

            bugs.append(bug)
    return bugs
def gather_session_summary(args: argparse.Namespace) -> SessionSummary:
    result_path = Path(args.result).expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    raw_results = load_json(result_path)
    contract_label, contract_section = infer_contract_section(raw_results, args.contract_label)

    max_chars = 20000

    result_text = result_path.read_text(encoding="utf-8", errors="ignore")
    if len(result_text) > max_chars:
        result_text = result_text[:max_chars] + "\n...[truncated]..."

    contract_source = None
    contract_file_path = Path(args.contract_file).expanduser()
    if contract_file_path.exists():
        contract_text = contract_file_path.read_text(encoding="utf-8", errors="ignore")
        if len(contract_text) > max_chars:
            contract_text = contract_text[:max_chars] + "\n...[truncated]..."
        contract_source = contract_text

    analysis_path = None
    analysis_text = None
    if args.analysis:
        analysis_candidate = Path(args.analysis).expanduser()
        if analysis_candidate.exists():
            analysis_path = str(analysis_candidate)
            analysis_content = analysis_candidate.read_text(encoding="utf-8", errors="ignore")
            if len(analysis_content) > max_chars:
                analysis_content = analysis_content[:max_chars] + "\n...[truncated]..."
            analysis_text = analysis_content

    log_path = None
    log_text = None
    if args.log:
        log_candidate = Path(args.log).expanduser()
        if log_candidate.exists():
            log_path = str(log_candidate)
            log_content = log_candidate.read_text(encoding="utf-8", errors="ignore")
            if len(log_content) > max_chars:
                log_content = log_content[-max_chars:]
            log_text = log_content

    return SessionSummary(
        contract_label=contract_label,
        contract_path=args.contract_file,
        result_path=str(result_path),
        result_json=raw_results,
        result_text=result_text,
        contract_source=contract_source,
        analysis_path=analysis_path,
        analysis_text=analysis_text,
        log_path=log_path,
        log_text=log_text,
        rag_storage_path=str(Path(args.rag_storage).expanduser()) if args.rag_storage else None,
        duplication_mode=args.duplication,
        fuzz_time=args.fuzz_time,
        max_trans_length=args.max_trans_length,
        provider_model=args.model,
        provider_name=args.provider,
        reporter_enabled=args.use_llm_summary,
        session_id=args.session_id or _utc_now().strftime("%Y%m%d%H%M%S"),
        threshold_high=getattr(args, "threshold_high", 70.0),
        threshold_low=getattr(args, "threshold_low", 30.0),
    )


def ensure_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def initialize_provider(args: argparse.Namespace) -> Optional[ProviderBase]:
    if not args.use_llm_summary:
        return None

    existing = provider_context.get_provider(optional=True)
    if existing:
        return existing

    model = args.model or config.get_default_llm_model()
    provider_name = args.provider or config.get_default_llm_provider()

    kwargs: Dict[str, Any] = {
        "ollama_endpoint": args.ollama_endpoint or config.get_default_ollama_endpoint(),
        "ollama_thinking": args.ollama_thinking or config.get_default_ollama_thinking(),
        "options": {"num_ctx": 8192},
    }

    normalized_provider = (provider_name or "").lower()

    # Chọn key theo provider:
    # - openai: dùng OPENAI_API_KEY (env) hoặc args.api_key
    # - gemini/google: dùng GOOGLE_API_KEY / config.get_google_api_key
    # - ollama: không cần key
    if normalized_provider == "openai":
        openai_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        provider = provider_factory.create_provider(
            model=model,
            provider_name=provider_name,
            openai_api_key=openai_key,
            default_provider=config.get_default_llm_provider(),
            **kwargs,
        )
    else:
        api_key = args.api_key or config.get_google_api_key()
        provider = provider_factory.create_provider(
            model=model,
            provider_name=provider_name,
            api_key=api_key,
            default_provider=config.get_default_llm_provider(),
            **kwargs,
        )
    provider_context.set_provider(provider)
    return provider


def build_llm_payload(summary: SessionSummary) -> Dict[str, Any]:
    """
    Build a compact, contract-centric payload for the LLM-as-a-judge.

    Focus on:
    - Contract source (for context)
    - Fuzzing metrics (coverage, generations, tx, time)
    - Flat list of bugs/errors to evaluate
    - High-level configuration (fuzz_time, max_trans_length, duplication)
    """
    contract_label = summary.contract_label
    raw_payload = summary.result_json if isinstance(summary.result_json, dict) else {}

    try:
        label, contract_section = infer_contract_section(raw_payload, contract_label)
    except Exception:
        # Fall back to an empty section; LLM should degrade gracefully.
        label, contract_section = contract_label, {}
        summary.technical_notes.append(
            f"Reporter could not locate dedicated contract section for '{contract_label}' in result.json"
        )

    metrics = _extract_metrics_from_contract_section(contract_section) if contract_section else {}
    bugs = _extract_bugs_from_contract_section(contract_section) if contract_section else []

    return {
        "session": {
            "id": summary.session_id,
            "timestamp": summary.timestamp.isoformat(),
            "fuzz_time": summary.fuzz_time,
            "max_trans_length": summary.max_trans_length,
            "duplication_mode": summary.duplication_mode,
            "provider_model": summary.provider_model,
            "provider_name": summary.provider_name,
        },
        "contract": {
            "label": label,
            "path": summary.contract_path,
            "source": summary.contract_source,
        },
        "metrics": metrics,
        "bugs": bugs,
        "artefacts": {
            "result_path": summary.result_path,
            "analysis_path": summary.analysis_path,
            "rag_storage_path": summary.rag_storage_path,
        },
        "technical_notes": summary.technical_notes,
    }


def _build_raw_artefacts(summary: SessionSummary) -> str:
    sections: List[str] = [
        "## Fuzzing Result JSON (excerpt)",
        "```json",
        summary.result_text.strip() if summary.result_text else "N/A",
        "```",
    ]

    if summary.contract_source:
        sections.extend(
            [
                "",
                "## Smart Contract Source (excerpt)",
                "```solidity",
                summary.contract_source.strip(),
                "```",
            ]
        )

    if summary.analysis_text:
        sections.extend(
            [
                "",
                "## Analysis File (excerpt)",
                "```",
                summary.analysis_text.strip(),
                "```",
            ]
        )

    if summary.log_text:
        sections.extend(
            [
                "",
                "## Log File (excerpt)",
                "```",
                summary.log_text.strip(),
                "```",
            ]
        )

    if summary.technical_notes:
        sections.extend(["", "## Technical Notes"])
        sections.extend(f"- {note}" for note in summary.technical_notes)

    sections.append("")
    return "\n".join(sections)


def build_fallback_report(summary: SessionSummary) -> str:
    sections = [
        f"# UniFuzz Fuzzing Report – Session {summary.session_id}",
        "",
        "Reporter LLM disabled or unavailable. Raw artefacts are included for manual review.",
        "",
        f"- Result file: `{summary.result_path}`",
        f"- Analysis file: `{summary.analysis_path or 'N/A'}`",
        f"- Log file: `{summary.log_path or 'N/A'}`",
        f"- RAG storage: `{summary.rag_storage_path or 'N/A'}`",
        "",
    ]

    sections.append(_build_raw_artefacts(summary))
    return "\n".join(sections)
def build_llm_prompt(summary: SessionSummary) -> str:
    payload = build_llm_payload(summary)
    return "INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\nEND_INPUT"


def parse_confidence_scores(llm_response: str, bugs: List[Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
    """
    Parse confidence scores and verdicts from LLM response.
    
    Args:
        llm_response: LLM response text
        bugs: List of bugs from payload (to map SWC ID to bug_id)
    
    Returns:
        Dict mapping bug_id -> (verdict, confidence_score)
        Example: {"96_0": ("CONFIRMED", 95.0)}
    """
    scores: Dict[str, Tuple[str, float]] = {}
    
    # Create mapping from SWC ID to bug_id
    swc_to_bug_id: Dict[int, str] = {}
    for bug in bugs:
        swc_id = bug.get("swc_id")
        bug_id = bug.get("id")
        if swc_id is not None and bug_id:
            swc_to_bug_id[swc_id] = bug_id
    
    # Pattern 1 (new format): [VERDICT] SWC-{id}: {type}. Conf {score}/100
    pattern1 = r'\[(CONFIRMED|REJECTED|INCONCLUSIVE)\].*?SWC-(\d+).*?Conf\s+(\d+(?:\.\d+)?)/100'
    # Pattern 2 (old format): [VERDICT] SWC-{id}: {type} (confidence: {score}/100)
    pattern2 = r'\[(CONFIRMED|REJECTED|INCONCLUSIVE)\].*?SWC-(\d+).*?\(confidence:\s*(\d+(?:\.\d+)?)/100\)'
    # Pattern 3 (old format variant): [VERDICT] SWC-{id}: {type} - {reason} (confidence: {score}/100)
    pattern3 = r'\[(CONFIRMED|REJECTED|INCONCLUSIVE)\].*?SWC-(\d+).*?confidence:\s*(\d+(?:\.\d+)?)/100'
    
    for pattern in [pattern1, pattern2, pattern3]:
        for match in re.finditer(pattern, llm_response, re.IGNORECASE | re.DOTALL):
            verdict = match.group(1).upper()
            swc_id = int(match.group(2))
            score = float(match.group(3))
            # Map SWC ID to bug_id
            if swc_id in swc_to_bug_id:
                bug_id = swc_to_bug_id[swc_id]
                # Only update if not already found (prefer first match)
                if bug_id not in scores:
                    scores[bug_id] = (verdict, score)
            else:
                # Fallback: use SWC ID as key
                fallback_key = f"swc_{swc_id}"
                if fallback_key not in scores:
                    scores[fallback_key] = (verdict, score)
    
    return scores


def apply_thresholds(
    confidence_score: float, threshold_high: float, threshold_low: float
) -> str:
    """
    Apply confidence thresholds to determine final verdict.
    
    Args:
        confidence_score: Confidence score from LLM (0-100)
        threshold_high: Threshold for CONFIRMED (>= this value)
        threshold_low: Threshold for REJECTED (< this value)
    
    Returns:
        Final verdict: "CONFIRMED", "REJECTED", or "INCONCLUSIVE"
    """
    if confidence_score >= threshold_high:
        return "CONFIRMED"
    elif confidence_score < threshold_low:
        return "REJECTED"
    else:
        return "INCONCLUSIVE"


def request_llm_summary(provider: ProviderBase, summary: SessionSummary) -> Optional[str]:
    try:
        prompt = build_llm_prompt(summary)
        system_prompt = build_reporter_system_prompt(
            threshold_high=summary.threshold_high,
            threshold_low=summary.threshold_low
        )
        # Log full prompt for transparency/debug
        LOGGER.info("[LLM Request] prompt=\n%s", prompt)
        LOGGER.info(
            "[LLM Request] Confidence thresholds: τ_high=%.1f, τ_low=%.1f (0-100 scale)",
            summary.threshold_high,
            summary.threshold_low
        )
        result: ProviderResult = provider.generate(
            prompt,
            system_prompt=system_prompt,
        )
        response_text = result.text.strip()
        
        # Parse confidence scores and apply thresholds
        payload = build_llm_payload(summary)
        bugs = payload.get("bugs", [])
        parsed_scores = parse_confidence_scores(response_text, bugs)
        if parsed_scores:
            LOGGER.info("[LLM Response] Parsed confidence scores: %s", parsed_scores)
            # Apply thresholds and update verdicts if needed
            for bug_key, (verdict, score) in parsed_scores.items():
                final_verdict = apply_thresholds(score, summary.threshold_high, summary.threshold_low)
                if final_verdict != verdict:
                    LOGGER.info(
                        "[LLM Response] Verdict adjusted for %s: %s -> %s (score=%.1f, thresholds=[τ_high=%.1f, τ_low=%.1f])",
                        bug_key, verdict, final_verdict, score, summary.threshold_high, summary.threshold_low
                    )
        else:
            LOGGER.warning("[LLM Response] No confidence scores parsed from response")
        
        return response_text
    except ProviderError as exc:
        message = f"Reporter provider failed: {exc}"
        LOGGER.warning(message)
        summary.technical_notes.append(message)
    except Exception as exc:
        message = f"Reporter provider unexpected failure: {exc}"
        LOGGER.warning(message)
        summary.technical_notes.append(message)
    return None


def update_progress_markdown(summary: SessionSummary, path: Path) -> None:
    contract_section = summary.result_json.get(summary.contract_label, {}) if isinstance(summary.result_json, dict) else {}

    coverage = None
    branch = None
    transactions = None
    unique_tx = None
    if isinstance(contract_section, dict):
        cov_entry = contract_section.get("code_coverage")
        if isinstance(cov_entry, dict):
            coverage = cov_entry.get("percentage")
        branch_entry = contract_section.get("branch_coverage")
        if isinstance(branch_entry, dict):
            branch = branch_entry.get("percentage")
        tx_entry = contract_section.get("transactions")
        if isinstance(tx_entry, dict):
            transactions = tx_entry.get("total")
            unique_tx = tx_entry.get("unique")

    section_lines = [
        f"### Báo cáo fuzzing {summary.contract_label} – {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Coverage: {coverage or 0:.2f}% / Branch: {branch or 0:.2f}%",
        f"- Transactions: {transactions or 0} (unique {unique_tx or 0})",
        "",
    ]

    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")

    updated = existing.rstrip() + "\n\n" + "\n".join(section_lines)
    path.write_text(updated, encoding="utf-8")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniFuzz reporter agent")
    parser.add_argument("--result", default=config.get_default_result_path())
    parser.add_argument("--analysis", default=str(ROOT_DIR / "dataflow_analysis_result.json"))
    parser.add_argument("--log", default=str(ROOT_DIR / "log" / "INFO.log"))
    parser.add_argument("--contract", dest="contract_spec", required=True, help="Format: path::ContractName")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rag-storage", default=str(ROOT_DIR / "RAG" / "rag_storage"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ollama-endpoint", default=None)
    parser.add_argument("--ollama-thinking", default=None)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--update-md", action="store_true")
    parser.add_argument("--progress-md", default=str(ROOT_DIR / "UNIFUZZ_PROGRESS_ANALYSIS.md"))
    parser.add_argument("--use-llm-summary", action="store_true")
    parser.add_argument("--fuzz-time", type=int, default=config.get_default_fuzz_time())
    parser.add_argument("--max-trans-length", type=int, default=config.get_default_max_trans_length())
    parser.add_argument("--duplication", default=config.get_default_duplication())
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=70.0,
        help="Confidence threshold for CONFIRMED verdict (0-100, default: 70.0). Bugs with confidence >= this value are CONFIRMED."
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=30.0,
        help="Confidence threshold for REJECTED verdict (0-100, default: 30.0). Bugs with confidence < this value are REJECTED."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def split_contract_spec(spec: str) -> Tuple[str, str]:
    if "::" not in spec:
        raise ValueError("Contract spec must follow '<path>::<ContractName>' format")
    path_part, label = spec.split("::", 1)
    return path_part.strip(), label.strip()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    contract_file, contract_label = split_contract_spec(args.contract_spec)
    args.contract_file = contract_file
    args.contract_label = contract_label

    try:
        summary = gather_session_summary(args)
    except Exception as exc:
        LOGGER.error("Failed to gather session summary: %s", exc)
        return 1

    provider = initialize_provider(args)
    llm_insight = request_llm_summary(provider, summary) if provider else None

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        contract_dir = ROOT_DIR / "reports" / _safe_slug(summary.contract_label or summary.contract_path)
        output_path = contract_dir / f"fuzz_report_{summary.session_id}.md"

    if llm_insight:
        # Chỉ lưu LLM response, không thêm raw artefacts hay prompt debug
        report_body = llm_insight.strip()
    else:
        report_body = build_fallback_report(summary)

    ensure_output_path(output_path)
    output_path.write_text(report_body, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)

    # In báo cáo trực tiếp ra terminal để tiện xem nhanh.
    print("\n" + "=" * 80 + "\nREPORT\n" + "=" * 80 + "\n")
    print(report_body)
    print("\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n")

    if args.update_md:
        progress_path = Path(args.progress_md).expanduser()
        ensure_output_path(progress_path)
        update_progress_markdown(summary, progress_path)
        LOGGER.info("Progress markdown updated at %s", progress_path)

    if provider_context.has_provider():
        provider_context.clear_provider()

    return 0


if __name__ == "__main__":
    sys.exit(main())

