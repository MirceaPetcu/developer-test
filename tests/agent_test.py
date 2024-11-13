import unittest
from agent.agent import Agent, LLMParams
import json


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = 'gpt-4o-mini'
        cls.agent = Agent(cls.model_id, eval=False)
        cls.eval_agent = Agent(cls.model_id, eval=True)

    def test_init(self):
        self.assertEqual(self.agent._model_id, self.model_id)
        self.assertEqual(self.agent._max_retries, 3)
        self.assertFalse(self.agent.eval)
        self.assertTrue(self.eval_agent.eval)
        self.assertIsNotNone(self.agent.logger)

    def test_get_messages(self):
        prompt = "Test prompt"
        system = "Test system"
        expected_messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        result = self.agent._get_messages(prompt, system)
        self.assertEqual(result, expected_messages)

    def test_call(self):
        prompt = "What is the capital of France?"
        system = "You are a helpful assistant."
        params = LLMParams(temperature=1, max_completion_tokens=100, top_p=1)

        result = self.agent(prompt, system, params)
        print(result)
        self.assertIsInstance(result, str)
        self.assertIn("Paris", result)

    def test_call_eval(self):
        prompt = "What is the capital of France?"
        system = "You are a helpful assistant. Respond in JSON format."
        params = LLMParams(temperature=1, max_completion_tokens=100, top_p=1)

        result = self.eval_agent(prompt, system, params)
        print(result)
        self.assertIsInstance(result, str)
        self.assertIn("Paris", result)
        self.assertIsInstance(json.loads(result), dict)

    def test_call_with_custom_params(self):
        prompt = "Write a short poem about spring."
        system = "You are a poetic assistant. Your poems are no longer than 60 words."
        params = LLMParams(temperature=0.5, max_completion_tokens=400, top_p=0.9)

        result = self.agent(prompt, system, params)
        print(result)
        self.assertIsInstance(result, str)
        self.assertLess(len(result.split()), 60)  # Roughly check if it's a short poem

    def test_call_with_error(self):
        prompt = "This is a test prompt."
        system = "You are a helpful assistant."
        params = LLMParams(max_completion_tokens=-1)  # Invalid token count

        with self.assertRaises(Exception):
            self.agent(prompt, system, params)


class TestLLMParams(unittest.TestCase):

    def test_default_values(self):
        params = LLMParams()
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.max_completion_tokens, 4096)
        self.assertEqual(params.top_p, 1.0)

    def test_custom_values(self):
        params = LLMParams(temperature=0.5, max_completion_tokens=1000, top_p=0.9)
        self.assertEqual(params.temperature, 0.5)
        self.assertEqual(params.max_completion_tokens, 1000)
        self.assertEqual(params.top_p, 0.9)


if __name__ == '__main__':
    unittest.main()
