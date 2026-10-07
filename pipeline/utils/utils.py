import re
import json

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
    result = {
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {}
    }
    
    if not data:
        return result

    if isinstance(data, dict):
        cost = data.get("total_cost_usd", 0.0)
        duration_ms = data.get("duration_api_ms", data.get("duration_ms", 0)) 
        
        requests = data.get("num_turns", 0)
        
        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        tokens_cached_read = usage.get("cache_read_input_tokens", 0)
        tokens_cached_created = usage.get("cache_creation_input_tokens", 0)
        
        cost_str = f"${float(cost):.2f}" if float(cost) > 0 else ""
        time_str = f"{duration_ms/1000:.2f}s" if duration_ms > 0 else ""
            
        result["requests"] = requests
        result["time_spent"] = time_str
        
        if requests > 0 or tokens_in > 0:
            str_in = format_number(tokens_in)
            str_out = format_number(tokens_out)
            
            cache_read_text = f", {format_number(tokens_cached_read)} cache read" if tokens_cached_read > 0 else ""
            cache_create_text = f", {format_number(tokens_cached_created)} cache created" if tokens_cached_created > 0 else ""
            
            req_word = "request" if requests == 1 else "requests"
            req_text = f" - {requests} {req_word}" if requests > 0 else ""
            
            result["models_breakdown"] = {
                model_name: f"{str_in} in, {str_out} out{cache_read_text}{cache_create_text} (Cost: {cost_str}{req_text})"
            }
            
        return result

    if isinstance(data, str):
        clean_log = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', data).lower()
            
        time_api = re.search(r'duration\s*\(api\)\s*:\s*([a-z0-9\.\s]+)', clean_log)
        if time_api: result["time_spent"] = time_api.group(1).strip()
            
        cost_match = re.search(r'cost\s*:\s*\$?([0-9\.]+)', clean_log)
        if cost_match:
            cost_val = float(cost_match.group(1))
            cost_str = f"${cost_val:.2f}"
        else:
            cost_str = ""

        turns_match = re.search(r'(\d+)\s+turns?', clean_log)
        requests = int(turns_match.group(1)) if turns_match else 0
        
        if cost_str or result["time_spent"] or requests > 0:
            result["requests"] = requests
            req_word = "request" if requests == 1 else "requests"
            req_text = f" - {requests} {req_word}" if requests > 0 else ""
            
            result["models_breakdown"] = {
                model_name: f"Cost: {cost_str}{req_text}".strip()
            }
            
    return result


def opencode_parse_stats(stdout: str, model_name: str) -> dict:
    result = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {}
    }

    if not stdout:
        return result

    total_input = 0
    total_output = 0
    total_cached = 0
    requests = 0

    for line in stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        if event.get("type") != "step_finish":
            continue

        part = event.get("part", {})
        tokens = part.get("tokens", {})
        cache = tokens.get("cache", {})

        total_input += tokens.get("input", 0)
        total_output += tokens.get("output", 0)
        total_cached += cache.get("read", 0)

        requests += 1

    result["requests"] = requests

    if requests > 0 or total_input > 0:
        req_word = "request" if requests == 1 else "requests"

        cached_text = (
            f", {format_number(total_cached)} cached"
            if total_cached > 0
            else ""
        )

        result["models_breakdown"] = {
            model_name: (
                f"{format_number(total_input)} in, "
                f"{format_number(total_output)} out"
                f"{cached_text} "
                f"(Est. {requests} {req_word})"
            )
        }

    return result


def codex_parse_stats(stdout: str, model_name: str) -> dict:
    result = {
        "changes_added": 0,
        "changes_removed": 0,
        "requests": 0,
        "time_spent": "",
        "models_breakdown": {},
    }

    if not stdout:
        return result

    total_input = 0
    total_cached = 0
    total_output = 0
    requests = 0

    for line in stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "turn.completed":
            usage = event.get("usage", {})

            total_input += usage.get("input_tokens", 0)
            total_cached += usage.get("cached_input_tokens", 0)
            total_output += usage.get("output_tokens", 0)

            requests += 1

    result["requests"] = requests

    if requests > 0 or total_input > 0:
        req_word = "request" if requests == 1 else "requests"

        cached_text = (
            f", {format_number(total_cached)} cached"
            if total_cached > 0
            else ""
        )

        result["models_breakdown"] = {
            model_name: (
                f"{format_number(total_input)} in, "
                f"{format_number(total_output)} out"
                f"{cached_text} "
                f"(Est. {requests} {req_word})"
            )
        }

    return result