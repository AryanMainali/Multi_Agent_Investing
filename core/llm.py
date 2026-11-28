"""Optional LLM client for Research (AI), Report (AI), and Monitor summary agents.

Supports OpenAI, Anthropic (including structured tool-schema JSON outputs), and Google
Gemini via standard HTTPS requests when an API key is configured, and falls back
cleanly to deterministic synthesis when no API key is present.
Never places orders or contacts any brokerage endpoint.
"""
from dataclasses import dataclass
import json
import os
import requests
from core.config import load_dotenv_file


@dataclass(frozen=True)
class LLMConfig:
    provider: str = 'auto'  # 'auto', 'openai', 'anthropic', 'gemini', or 'none'
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 20.0


DEFAULT_MODELS = {
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-haiku-latest',
    'gemini': 'gemini-2.0-flash',
}


def resolve_provider_and_key(config: LLMConfig | None = None) -> tuple[str | None, str | None, str | None]:
    load_dotenv_file()
    cfg = config or LLMConfig()
    if cfg.provider == 'none':
        return None, None, None

    anthropic_default = (os.environ.get('CLAUDE_MODEL') or '').strip() or DEFAULT_MODELS['anthropic']
    defaults = {**DEFAULT_MODELS, 'anthropic': anthropic_default}

    if cfg.provider in ('openai', 'anthropic', 'gemini'):
        env_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY',
        }
        key = (cfg.api_key or os.environ.get(env_map[cfg.provider], '')).strip()
        if not key:
            return None, None, None
        return cfg.provider, key, cfg.model or defaults[cfg.provider]

    for provider, env_var in (
        ('openai', 'OPENAI_API_KEY'),
        ('anthropic', 'ANTHROPIC_API_KEY'),
        ('gemini', 'GEMINI_API_KEY'),
    ):
        key = (cfg.api_key or os.environ.get(env_var, '')).strip()
        if key:
            return provider, key, cfg.model or defaults[provider]

    return None, None, None


def generate_text(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig | None = None,
    max_tokens: int = 900,
) -> tuple[str | None, str]:
    """Call configured LLM provider and return (text, mode_label), or (None, fallback_reason)."""
    provider, key, model = resolve_provider_and_key(config)
    if not provider or not key or not model:
        return None, 'deterministic_synthesis (no LLM API key configured)'

    timeout = config.timeout_seconds if config else 20.0
    try:
        if provider == 'openai':
            resp = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                json={
                    'model': model,
                    'temperature': 0.2,
                    'max_tokens': max_tokens,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content'].strip()
            return text, f'llm:openai:{model}'

        if provider == 'anthropic':
            resp = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model,
                    'max_tokens': max_tokens,
                    'temperature': 0.2,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            blocks = resp.json().get('content', [])
            text = '\n'.join(b.get('text', '') for b in blocks if b.get('type') == 'text').strip()
            return (text or None), f'llm:anthropic:{model}'

        if provider == 'gemini':
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            resp = requests.post(
                url,
                params={'key': key},
                headers={'Content-Type': 'application/json'},
                json={
                    'systemInstruction': {'parts': [{'text': system_prompt}]},
                    'contents': [{'parts': [{'text': user_prompt}]}],
                    'generationConfig': {'temperature': 0.2},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            parts = resp.json()['candidates'][0]['content']['parts']
            text = '\n'.join(p.get('text', '') for p in parts).strip()
            return (text or None), f'llm:gemini:{model}'

    except Exception as exc:
        return None, f'deterministic_synthesis (LLM fallback: {type(exc).__name__})'

    return None, 'deterministic_synthesis'


def call_structured_tool(
    system_prompt: str,
    user_prompt: str,
    tool_schema: dict,
    config: LLMConfig | None = None,
    max_tokens: int = 1200,
) -> tuple[dict | None, str]:
    """Invoke Claude forced tool_choice JSON schema (or JSON prompt fallback on OpenAI/Gemini)."""
    provider, key, model = resolve_provider_and_key(config)
    if not provider or not key or not model:
        return None, 'deterministic_keywords'

    timeout = config.timeout_seconds if config else 20.0
    if provider == 'anthropic':
        try:
            resp = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model,
                    'max_tokens': max_tokens,
                    'temperature': 0.1,
                    'system': system_prompt,
                    'tools': [tool_schema],
                    'tool_choice': {'type': 'tool', 'name': tool_schema['name']},
                    'messages': [{'role': 'user', 'content': user_prompt}],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            for block in resp.json().get('content', []):
                if block.get('type') == 'tool_use' and isinstance(block.get('input'), dict):
                    return block['input'], f'llm_tool:anthropic:{model}'
        except Exception as exc:
            return None, f'deterministic_keywords (fallback: {type(exc).__name__})'

    schema_prompt = (
        f"{user_prompt}\n\nReturn ONLY valid JSON conforming to this schema:\n"
        f"{json.dumps(tool_schema.get('input_schema', {}))}"
    )
    raw, mode = generate_text(system_prompt, schema_prompt, config=config, max_tokens=max_tokens)
    parsed = parse_json_object(raw)
    return parsed, mode


def parse_json_object(raw_text: str | None) -> dict | None:
    """Safely extract a JSON object from an LLM response (including fenced blocks)."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith('```')]
        cleaned = '\n'.join(lines).strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
