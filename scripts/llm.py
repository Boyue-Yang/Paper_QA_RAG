"""DeepSeek chat client used by every LLM step in the pipeline (drafting, verifying, revising, judging).

The pipeline asks for a role ("drafter", "verifier", "controller", "judge"); ROLES maps each to a model and thinking
setting, so changing a model is a one-line edit here. The API key comes from the DEEPSEEK_API_KEY
environment variable or a gitignored `.env` file in the project root, never from code.

  python -m scripts.llm ping            # one cheap call per role to check key and settings
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://api.deepseek.com'
FLASH = 'deepseek-flash'
PRO = 'deepseek-v4-pro'
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ModelSpec:
    model: str
    thinking: bool
    reasoning_effort: str = 'high'  # low | high | max; only sent when thinking is on
    temperature: float = 0.0        # ignored by the API in thinking mode, so only sent when thinking is off
    max_tokens: int = 4096          # in thinking mode this also has to cover the reasoning


ROLES = {
    'drafter': ModelSpec(FLASH, thinking=False, max_tokens=4096),
    'verifier': ModelSpec(FLASH, thinking=True, reasoning_effort='high', max_tokens=16384),
    'controller': ModelSpec(FLASH, thinking=True, reasoning_effort='high', max_tokens=16384),
    'judge': ModelSpec(PRO, thinking=True, reasoning_effort='high', max_tokens=16384),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: str
    reasoning: str
    usage: dict
    model: str
    finish_reason: str
    cached: bool = False
    parsed: object = None


def load_api_key(env_file=ROOT / '.env'):
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding='utf-8').splitlines():
            name, _, value = line.partition('=')
            if name.strip() == 'DEEPSEEK_API_KEY':
                key = value.strip().strip('"').strip("'")
    if not key:
        raise LLMError('DEEPSEEK_API_KEY is not set. Put DEEPSEEK_API_KEY=... in the environment '
                       f'or in {env_file} (this file is gitignored).')
    return key


class LLM:
    def __init__(self, role='drafter', spec=None, api_key=None, session=None, cache_dir=None,
                 log_path=None, max_retries=4, timeout=300):
        self.role = role
        self.spec = spec or ROLES[role]
        self._api_key = api_key
        self.session = session or requests.Session()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.log_path = Path(log_path) if log_path else None
        self.max_retries = max_retries
        self.timeout = timeout

    def payload(self, system, user, json_mode=False):
        s = self.spec
        body = {'model': s.model, 'max_tokens': s.max_tokens,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if s.thinking:
            body['thinking'] = {'type': 'enabled'}
            body['reasoning_effort'] = s.reasoning_effort
        else:
            body['thinking'] = {'type': 'disabled'}
            body['temperature'] = s.temperature
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        return body

    def chat(self, system, user, json_mode=False):
        """One completion. With json_mode the reply is parsed into result.parsed (retried if it is blank or invalid).

        The prompt must itself contain the word "json" and an example of the shape wanted.
        """
        if json_mode and 'json' not in (system + user).lower():
            raise LLMError('json_mode needs the word "json" and an example of the output shape in the prompt')
        body = self.payload(system, user, json_mode)
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        cached = self._read_cache(key)
        if cached:
            self._log(key, cached, 0.0, cached=True)
            return cached
        started = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                result = self._post(body)
                if json_mode:
                    result.parsed = json.loads(result.content)  # blank or truncated content raises here
                break
            except (requests.RequestException, json.JSONDecodeError, _Retry) as error:
                if attempt == self.max_retries:
                    raise LLMError(f'{self.role}: gave up after {attempt + 1} attempts: {error}') from error
                time.sleep(min(2 ** attempt, 30))
        self._write_cache(key, result)
        self._log(key, result, time.time() - started, cached=False)
        return result

    def _post(self, body):
        response = self.session.post(
            BASE_URL + '/chat/completions', json=body, timeout=self.timeout,
            headers={'Authorization': 'Bearer ' + (self._api_key or load_api_key())})
        if response.status_code in RETRY_STATUS:
            raise _Retry(f'HTTP {response.status_code}')
        if response.status_code != 200:
            # not retryable (bad key, bad request, no balance): report the server's message, never the key
            raise LLMError(f'{self.role}: HTTP {response.status_code}: {response.text[:300]}')
        data = response.json()
        choice = data['choices'][0]
        message = choice['message']
        return LLMResult(content=message.get('content') or '', reasoning=message.get('reasoning_content') or '',
                         usage=data.get('usage', {}), model=data.get('model', body['model']),
                         finish_reason=choice.get('finish_reason', ''))

    def _read_cache(self, key):
        if not self.cache_dir:
            return None
        path = self.cache_dir / (key + '.json')
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        return LLMResult(**{**data, 'cached': True})

    def _write_cache(self, key, result):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data = {k: getattr(result, k) for k in ['content', 'reasoning', 'usage', 'model', 'finish_reason', 'parsed']}
            (self.cache_dir / (key + '.json')).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    def _log(self, key, result, seconds, cached):
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'role': self.role, 'model': result.model,
                  'request': key[:16], 'cached': cached, 'seconds': round(seconds, 2),
                  'finish_reason': result.finish_reason, 'usage': result.usage}
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


class _Retry(Exception):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['ping'])
    parser.parse_args(argv)
    for role in ROLES:
        llm = LLM(role)
        result = llm.chat('You are a terse assistant.', 'Reply with exactly: ok')
        print(f"{role:9s} {result.model:28s} finish={result.finish_reason} "
              f"reply={result.content.strip()[:20]!r} usage={result.usage}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
