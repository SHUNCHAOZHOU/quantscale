"""Conservative, auditable normalization of manager assignments."""
import json
import re


class ManagerProtocolError(ValueError):
    pass


def strict_json_object(content):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(content, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def normalize_worker_ids(directions, fleet_size):
    if not isinstance(directions, dict) or any(not isinstance(k, str) for k in directions):
        return directions
    expected = {str(i) for i in range(fleet_size)}
    if set(directions) == expected:
        return directions
    stripped = {key.removeprefix("worker_"): value for key, value in directions.items()}
    if len(stripped) != len(directions):
        return directions
    if set(stripped) == expected:
        return stripped
    if set(stripped) == {str(i + 1) for i in range(fleet_size)}:
        return {str(i): stripped[str(i + 1)] for i in range(fleet_size)}
    return directions


def parse_manager(response, fleet_size, families=None):
    """Return assignments and explicit transformations; never invent missing tasks."""
    if not isinstance(response, dict) or "adapter_error" in response:
        raise ManagerProtocolError("Manager response is not a complete JSON object")
    if any(not isinstance(key, str) for key in response):
        raise ManagerProtocolError("Manager keys must be strings")
    source = {k: v for k, v in response.items() if k != "_provenance"}
    actions = []
    wrapped = "directions" in source
    directions = source.get("directions", source)
    if not isinstance(directions, dict):
        raise ManagerProtocolError("directions must be an object")
    if any(not isinstance(key, str) for key in directions):
        raise ManagerProtocolError("Worker keys must be strings")
    if not wrapped:
        actions.append("added_directions_wrapper")
    elif any(re.fullmatch(r"(?:worker_)?\d+", k) for k in source if k != "directions"):
        raise ManagerProtocolError("Conflicting wrapped and top-level worker assignments")
    guides = []
    for obj in ([source, directions] if wrapped else [directions]):
        if "cross_cutting_guidance" in obj:
            guide = obj["cross_cutting_guidance"]
            if not (isinstance(guide, str) and guide.strip()
                    or isinstance(guide, dict) and guide
                    and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in guide.items())):
                raise ManagerProtocolError("Ambiguous shared guidance")
            if guide not in guides:
                guides.append(guide)
    if len(guides) > 1:
        raise ManagerProtocolError("Conflicting shared guidance")
    root_extras = set(source) - {"directions", "cross_cutting_guidance", "type"} if wrapped else set()
    if root_extras:
        raise ManagerProtocolError("Unexpected manager fields: " + ", ".join(sorted(root_extras)))
    if "type" in source and source["type"] != "json_object":
        raise ManagerProtocolError("Unsupported manager type")
    workers = {k: v for k, v in directions.items() if k not in ("cross_cutting_guidance", "type")}
    normalized = normalize_worker_ids(workers, fleet_size)
    if set(normalized) != {str(i) for i in range(fleet_size)}:
        raise ManagerProtocolError("Manager must assign every worker exactly once with unambiguous IDs")
    if set(normalized) != set(workers):
        actions.append("normalized_worker_ids")
    result = {}
    for key, value in normalized.items():
        if isinstance(value, dict):
            fields = set(value) & {"assignment", "task", "direction"}
            if len(fields) != 1 or set(value) - fields - {"theme", "constraints_note", "notes", "rationale"}:
                raise ManagerProtocolError("Ambiguous nested assignment for worker " + key)
            primary = next(iter(fields))
            if any(not isinstance(v, str) or not v.strip() for v in value.values()):
                raise ManagerProtocolError("Nested assignment fields must be nonempty strings")
            task = value[primary]
            for label in sorted(set(value) - {primary}):
                task += "\n" + label + ": " + value[label]
            value = task
            actions.append("flattened_assignment:" + key)
        if not isinstance(value, str) or not value.strip():
            raise ManagerProtocolError("Manager assignment must be a nonempty string for worker " + key)
        if families is not None and value not in families:
            raise ManagerProtocolError("Rules mode assignment must be an allowed family")
        result[key] = value
    if guides:
        if families is not None:
            raise ManagerProtocolError("Shared text guidance is unsupported in rules mode")
        shared = guides[0] if isinstance(guides[0], str) else json.dumps(guides[0], ensure_ascii=False, sort_keys=True)
        result = {k: v + "\n\nShared manager guidance:\n" + shared for k, v in result.items()}
        actions.append("preserved_shared_guidance")
    return result, actions


def repair_reference(response, fleet_size):
    """A repair may only reformat a complete set of verbatim simple assignments.

    On malformed JSON, recover direct worker/string pairs for comparison only.
    Duplicates, unknown IDs and complex ambiguous content never authorize guessing.
    """
    if (not isinstance(response, dict) or response.get("adapter_error_kind") != "json_parse"
            or response.get("finish_reason") != "stop"):
        return None
    content = response.get("raw_content", "")
    if not isinstance(content, str) or not content:
        return None
    pattern = r'(?<!\\)("(?:worker_)?\d+")\s*:\s*("(?:[^"\\]|\\.)*")'
    pairs = re.findall(pattern, content)
    keys = [json.loads(k) for k, _ in pairs]
    if len(keys) != fleet_size or len(set(keys)) != fleet_size:
        return None
    # Only punctuation/packaging errors in simple assignment maps qualify.
    # Complex or shared instructions must not silently disappear in a repair.
    string_pattern = r'"(?:[^"\\]|\\.)*"'
    try:
        tokens = [json.loads(token) for token in re.findall(string_pattern, content)]
        allowed = set(keys) | {json.loads(v) for _, v in pairs} | {"directions"}
    except ValueError:
        return None
    skeleton = re.sub(string_pattern, '""', content)
    if (any(token not in allowed for token in tokens) or tokens.count("directions") > 1
            or re.search(r'[^\s{}:,\"]', skeleton)):
        return None
    try:
        expected, _ = parse_manager({json.loads(k): json.loads(v) for k, v in pairs}, fleet_size)
    except (ValueError, TypeError):
        return None
    return expected
