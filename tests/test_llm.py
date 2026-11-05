"""LLM client behaviour checked against a fake HTTP session: no network, no key."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import llm
from scripts.llm import LLM, LLMError, ROLES, load_api_key


class FakeResponse:
    def __init__(self, status=200, content='ok', reasoning=None):
        self.status_code = status
        self.text = content
        message = {'content': content}
        if reasoning:
            message['reasoning_content'] = reasoning
        self._data = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 5, 'completion_tokens': 2},
                      'choices': [{'message': message, 'finish_reason': 'stop'}]}

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({'url': url, 'json': json, 'headers': headers})
        return self.responses.pop(0)


def client(*responses, **kwargs):
    session = FakeSession(*responses)
    kwargs.setdefault('api_key', 'test-key')
    return LLM('drafter', session=session, **kwargs), session


class LLMTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(llm.time, 'sleep')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_thinking_off_sends_temperature_and_disabled_flag(self):
        model, session = client(FakeResponse())
        model.chat('sys', 'hello')
        body = session.calls[0]['json']
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['temperature'], 0.0)
        self.assertNotIn('reasoning_effort', body)
        self.assertEqual(session.calls[0]['headers']['Authorization'], 'Bearer test-key')

    def test_thinking_on_sends_effort_and_returns_reasoning(self):
        session = FakeSession(FakeResponse(content='answer', reasoning='because'))
        model = LLM('verifier', session=session, api_key='k')
        result = model.chat('sys', 'hello')
        body = session.calls[0]['json']
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['reasoning_effort'], ROLES['verifier'].reasoning_effort)
        self.assertNotIn('temperature', body)
        self.assertEqual((result.content, result.reasoning), ('answer', 'because'))

    def test_retries_rate_limit_then_succeeds(self):
        model, session = client(FakeResponse(429), FakeResponse(503), FakeResponse(content='fine'))
        self.assertEqual(model.chat('s', 'u').content, 'fine')
        self.assertEqual(len(session.calls), 3)

    def test_gives_up_after_max_retries(self):
        model, session = client(*[FakeResponse(500) for _ in range(3)], max_retries=2)
        with self.assertRaises(LLMError):
            model.chat('s', 'u')
        self.assertEqual(len(session.calls), 3)

    def test_auth_error_is_not_retried_and_hides_key(self):
        model, session = client(FakeResponse(401, content='Authentication Fails'))
        with self.assertRaises(LLMError) as caught:
            model.chat('s', 'u')
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn('test-key', str(caught.exception))

    def test_json_mode_retries_blank_reply_and_parses(self):
        model, session = client(FakeResponse(content='   '), FakeResponse(content='{"a": 1}'))
        result = model.chat('Reply in json like {"a": 0}', 'u', json_mode=True)
        self.assertEqual(result.parsed, {'a': 1})
        self.assertEqual(session.calls[1]['json']['response_format'], {'type': 'json_object'})

    def test_json_mode_requires_json_in_prompt(self):
        model, session = client()
        with self.assertRaises(LLMError):
            model.chat('s', 'u', json_mode=True)
        self.assertEqual(session.calls, [])

    def test_cache_skips_second_call_and_log_records_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'calls.jsonl'
            model, session = client(FakeResponse(content='once'), cache_dir=Path(folder) / 'cache', log_path=log)
            first = model.chat('s', 'u')
            second = model.chat('s', 'u')
            self.assertEqual((first.content, second.content, second.cached), ('once', 'once', True))
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(len(log.read_text(encoding='utf-8').splitlines()), 2)

    def test_api_key_from_env_or_dotenv_and_helpful_error(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / '.env'
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(LLMError) as caught:
                    load_api_key(env_file)
                self.assertIn('DEEPSEEK_API_KEY', str(caught.exception))
                env_file.write_text('OTHER=1\nDEEPSEEK_API_KEY="from-file"\n', encoding='utf-8')
                self.assertEqual(load_api_key(env_file), 'from-file')
            with mock.patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'from-env'}):
                self.assertEqual(load_api_key(env_file), 'from-env')


if __name__ == '__main__':
    unittest.main()
