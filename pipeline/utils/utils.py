import re
import json

try:
    from .generation_usage import parse_generation_usage
except ImportError:  # loaded without package context
    from utils.generation_usage import parse_generation_usage


def _display_num(value) -> str:
    """Transitional display helper: numerics via format_number, null as unknown."""
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return format_number(value)
    return "unknown"


def _render_est_breakdown(model_name: str, usage: dict) -> dict:
    """Shared Codex/OpenCode legacy breakdown: display only, numerics stay null."""
    requests = usage.get("requests") or 0
    total_input = usage.get("input_tokens")
    total_output = usage.get("output_tokens")
    total_cached = usage.get("cache_read_tokens")

    if not (requests > 0 or (isinstance(total_input, int) and total_input > 0)):
        return {"requests": requests, "time_spent": "", "models_breakdown": {}}

    req_word = "request" if requests == 1 else "requests"
    cached_text = (
        f", {format_number(total_cached)} cached"
        if isinstance(total_cached, int) and total_cached > 0
        else ""
    )
    return {
        "requests": requests,
        "time_spent": "",  # adapters own wall timing; no trusted CLI duration
        "models_breakdown": {
            model_name: (
                f"{_display_num(total_input)} in, "
                f"{_display_num(total_output)} out"
                f"{cached_text} "
                f"(Est. {requests} {req_word})"
            )
        },
    }

def format_number(num: int) -> str:
    if num >= 1000000:
        return f"{num/1000000:.1f}m".replace('.0m', 'm')
    elif num >= 1000:
        return f"{num/1000:.1f}k".replace('.0k', 'k')
    return str(num)


def gemini_parse_stats(stdout: str, model_name: str) -> dict:
    result = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {}
    }

    if not stdout:
        return result

    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return result

    usage = data.get("usage", {})

    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    tokens_cached = usage.get("cache_read_tokens", 0)

    requests = data.get("num_turns", 0)
    duration = data.get("duration_seconds", 0)

    result["requests"] = requests
    result["time_spent"] = (
        f"{duration:.2f}s"
        if duration > 0
        else ""
    )

    if requests > 0 or tokens_in > 0:
        req_word = "request" if requests == 1 else "requests"

        cached_text = (
            f", {format_number(tokens_cached)} cached"
            if tokens_cached > 0
            else ""
        )

        result["models_breakdown"] = {
            model_name: (
                f"{format_number(tokens_in)} in, "
                f"{format_number(tokens_out)} out"
                f"{cached_text} "
                f"(Est. {requests} {req_word})"
            )
        }

    return result


def copilot_parse_stats(log_text: str, model_name: str) -> dict:
    stats = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
    }

    if not log_text:
        return stats

    # ---------------------------------------------------------
    # Changes
    # Example:
    # Changes    +13 -0
    # ---------------------------------------------------------
    changes_match = re.search(
        r"Changes\s+\+(\d+)\s+-\s*(\d+)",
        log_text,
        re.IGNORECASE,
    )

    if changes_match:
        stats["changes_added"] = int(changes_match.group(1))
        stats["changes_removed"] = int(changes_match.group(2))

    # ---------------------------------------------------------
    # Time spent
    # Example:
    # AI Credits 1.29
    # Tokens ... 
    # Resume copilot --resume=... 
    #
    # The duration appears in the agent message:
    # "6s"
    # "2m 8s"
    # ---------------------------------------------------------
    time_matches = re.findall(
        r"\b(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s\b",
        log_text,
    )

    if time_matches:
        minutes, seconds = time_matches[-1]

        if minutes:
            stats["time_spent"] = f"{minutes}m {seconds}s"
        else:
            stats["time_spent"] = f"{seconds}s"

    # ---------------------------------------------------------
    # Tokens
    # Example:
    # Tokens     ↑ 544.8k (512.3k cached, 32.5k written)
    #            ↓ 7.5k (4.9k reasoning)
    # ---------------------------------------------------------
    token_match = re.search(
        r"Tokens\s+"
        r"↑\s*([\d.]+[kKmM]?)\s*"
        r"\(([\d.]+[kKmM]?)\s+cached,\s*([\d.]+[kKmM]?)\s+written\)"
        r"\s*•\s*"
        r"↓\s*([\d.]+[kKmM]?)"
        r"\s*\(([\d.]+[kKmM]?)\s+reasoning\)",
        log_text,
        re.IGNORECASE,
    )

    if token_match:
        raw_in = token_match.group(1).lower()
        raw_cached = token_match.group(2).lower()
        raw_written = token_match.group(3).lower()
        raw_out = token_match.group(4).lower()
        raw_reasoning = token_match.group(5).lower()

        stats["models_breakdown"] = {
            model_name: (
                f"{raw_in} in, "
                f"{raw_out} out, "
                f"{raw_cached} cached, "
                f"{raw_written} written, "
                f"{raw_reasoning} reasoning"
            )
        }

    return stats


def claude_parse_stats(data, model_name: str) -> dict:
    """Legacy-shape wrapper over the numeric parser (transitional).

    Keeps the historical return keys so existing adapters work unchanged
    during phase 1. Numeric truth lives in generation_usage; the
    models_breakdown string here is display only.
    """
    result = {
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {}
    }

    if not data:
        return result

    if not isinstance(data, (dict, str)):
        return result

    parsed = parse_generation_usage("claude", data, model=model_name)

    usage = parsed["usage"]
    requests = usage.get("requests") or 0
    tokens_in = usage.get("input_tokens")
    tokens_out = usage.get("output_tokens")

    result["requests"] = requests
    result["time_spent"] = ""  # no trusted CLI duration; adapters own timing

    if requests > 0 or (isinstance(tokens_in, int) and tokens_in > 0):
        cost = usage.get("cost_usd")
        cost_str = f"${float(cost):.2f}" if isinstance(cost, (int, float)) else ""

        tokens_cached_read = usage.get("cache_read_tokens")
        tokens_cached_created = usage.get("cache_write_tokens")
        cache_read_text = (
            f", {format_number(tokens_cached_read)} cache read"
            if isinstance(tokens_cached_read, int) and tokens_cached_read > 0
            else ""
        )
        cache_create_text = (
            f", {format_number(tokens_cached_created)} cache created"
            if isinstance(tokens_cached_created, int) and tokens_cached_created > 0
            else ""
        )

        req_word = "request" if requests == 1 else "requests"
        req_text = f" - {requests} {req_word}" if requests > 0 else ""

        result["models_breakdown"] = {
            model_name: (
                f"{_display_num(tokens_in)} in, {_display_num(tokens_out)} out"
                f"{cache_read_text}{cache_create_text} (Cost: {cost_str}{req_text})"
            )
        }

    return result


def opencode_parse_stats(stdout: str, model_name: str) -> dict:
    """Legacy-shape wrapper over the numeric parser (transitional).

    Keeps the historical return keys so the existing adapter works unchanged
    during phase 1. The adapter still overwrites time_spent with its own
    wall measurement; numeric truth lives in generation_usage.
    """
    result = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {}
    }

    if not stdout:
        return result

    if not isinstance(stdout, (dict, str)):
        return result

    parsed = parse_generation_usage("opencode", stdout, model=model_name)
    result.update(_render_est_breakdown(model_name, parsed["usage"]))

    return result


def codex_parse_stats(stdout: str, model_name: str) -> dict:
    """Legacy-shape wrapper over the numeric parser (transitional).

    Keeps the historical return keys so the existing adapter works unchanged
    during phase 1. The adapter still overwrites time_spent with its own
    wall measurement; numeric truth lives in generation_usage.
    """
    result = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {},
    }

    if not stdout:
        return result

    if not isinstance(stdout, (dict, str)):
        return result

    parsed = parse_generation_usage("codex", stdout, model=model_name)
    result.update(_render_est_breakdown(model_name, parsed["usage"]))

    return result