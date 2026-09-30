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
        stats = data.get("stats", {})
        
        files = stats.get("files", {})
        result["changes_added"] = files.get("totalLinesAdded", 0)
        result["changes_removed"] = files.get("totalLinesRemoved", 0)
        
        model_data = stats.get("models", {}).get(model_name, {})
        
        tokens_in = model_data.get("tokens", {}).get("input", 0)
        tokens_out = model_data.get("tokens", {}).get("candidates", 0)
        tokens_cached = model_data.get("tokens", {}).get("cached", 0)
        
        requests = model_data.get("api", {}).get("totalRequests", 0)
        latency_ms = model_data.get("api", {}).get("totalLatencyMs", 0)
        latency_s = latency_ms / 1000
        
        result["requests"] = requests
        result["time_spent"] = f"{latency_s:.2f}s" if latency_s > 0 else ""

        if requests > 0 or tokens_in > 0:
            req_word = "request" if requests == 1 else "requests"
            
            str_in = format_number(tokens_in)
            str_out = format_number(tokens_out)
            str_cached = format_number(tokens_cached)
            
            cached_text = f", {str_cached} cached" if tokens_cached > 0 else ""
            
            result["models_breakdown"] = {
                model_name: f"{str_in} in, {str_out} out{cached_text} (Est. {requests} {req_word})"
            }

        return result
    except Exception:
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

    raw_in = "0"
    raw_out = "0"
    raw_cached = "0"

    changes_match = re.search(r'Changes\s+\+([\d]+)\s+\-([\d]+)', log_text)
    if changes_match:
        stats["changes_added"] = int(changes_match.group(1))
        stats["changes_removed"] = int(changes_match.group(2))

    req_match = re.search(r'Requests\s+(\d+).*?\(([^)]+)\)', log_text)
    if req_match:
        stats["requests"] = int(req_match.group(1))
        stats["time_spent"] = req_match.group(2).strip()
        
    in_match = re.search(r'↑\s*([\d\.]+[kmKM]?)', log_text)
    if in_match:
        raw_in = in_match.group(1).lower()
        
    out_match = re.search(r'↓\s*([\d\.]+[kmKM]?)', log_text)
    if out_match:
        raw_out = out_match.group(1).lower()
        
    cache_match = re.search(r'([\d\.]+[kmKM]?)\s*\(cached\)', log_text)
    if cache_match:
        raw_cached = cache_match.group(1).lower()

    if stats["requests"] > 0 or raw_in != "0":
        req_word = "request" if stats["requests"] == 1 else "requests"
        
        stats["models_breakdown"] = {
            f"{model_name}": f"{raw_in} in, {raw_out} out, {raw_cached} cached (Est. {stats['requests']} Premium {req_word})"
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

    try:
        data = json.loads(stdout)
        
        stats = data.get("stats", {})
        
        files = stats.get("files", {})
        result["changes_added"] = files.get("totalLinesAdded", 0)
        result["changes_removed"] = files.get("totalLinesRemoved", 0)
        
        model_data = stats.get("models", {}).get(model_name, {})
        
        tokens_in = model_data.get("tokens", {}).get("input", 0)
        tokens_out = model_data.get("tokens", {}).get("candidates", 0)
        tokens_cached = model_data.get("tokens", {}).get("cached", 0)
        
        requests = model_data.get("api", {}).get("totalRequests", 0)
        latency_ms = model_data.get("api", {}).get("totalLatencyMs", 0)
        latency_s = latency_ms / 1000
        
        result["requests"] = requests
        result["time_spent"] = f"{latency_s:.2f}s" if latency_s > 0 else ""

        if requests > 0 or tokens_in > 0:
            req_word = "request" if requests == 1 else "requests"
            
            str_in = format_number(tokens_in)
            str_out = format_number(tokens_out)
            str_cached = format_number(tokens_cached)
            
            cached_text = f", {str_cached} cached" if tokens_cached > 0 else ""
            
            result["models_breakdown"] = {
                model_name: f"{str_in} in, {str_out} out{cached_text} (Est. {requests} {req_word})"
            }

        return result
    except Exception:
        return result


def codex_parse_stats(stdout: str, model_name: str) -> dict:
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
        
        stats = data.get("stats", {})
        
        files = stats.get("files", {})
        result["changes_added"] = files.get("totalLinesAdded", 0)
        result["changes_removed"] = files.get("totalLinesRemoved", 0)
        
        model_data = stats.get("models", {}).get(model_name, {})
        
        tokens_in = model_data.get("tokens", {}).get("input", 0)
        tokens_out = model_data.get("tokens", {}).get("candidates", 0)
        tokens_cached = model_data.get("tokens", {}).get("cached", 0)
        
        requests = model_data.get("api", {}).get("totalRequests", 0)
        latency_ms = model_data.get("api", {}).get("totalLatencyMs", 0)
        latency_s = latency_ms / 1000
        
        result["requests"] = requests
        result["time_spent"] = f"{latency_s:.2f}s" if latency_s > 0 else ""

        if requests > 0 or tokens_in > 0:
            req_word = "request" if requests == 1 else "requests"
            
            str_in = format_number(tokens_in)
            str_out = format_number(tokens_out)
            str_cached = format_number(tokens_cached)
            
            cached_text = f", {str_cached} cached" if tokens_cached > 0 else ""
            
            result["models_breakdown"] = {
                model_name: f"{str_in} in, {str_out} out{cached_text} (Est. {requests} {req_word})"
            }

        return result
    except Exception:
        return result