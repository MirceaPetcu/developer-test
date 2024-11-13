import logging
from openai import AzureOpenAI
import os
from typing import List
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from utils.utils import get_logger_file_handler


class LLMParams(BaseModel):
    """
    Parameters for the language model.
    """
    temperature: float = 1.0
    max_completion_tokens: int = 4096
    top_p: float = 1.0


class Agent:
    def __init__(self, model_id: str, eval: bool = True) -> None:
        self._model_id = model_id
        self._client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
        )
        self._max_retries = 3
        self.eval = eval
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(get_logger_file_handler("eval_agent.log" if eval else "agent.log"))
        self.logger.info(f"Agent initialized with model_id: {model_id}")

    def _get_messages(self, prompt: str, system: str) -> List[dict]:
        """
        Creates the messages to send to the API.
        :param prompt: user prompt
        :param system: system prompt
        :return: the messages
        """
        self.logger.debug(f"Creating messages with prompt: {prompt[:50]}... and system: {system[:50]}...")
        return [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def __call__(self, prompt: str, system: str, params: LLMParams) -> str:
        """
        Calls the agent with the given prompt.
        :param prompt: user prompt
        :param system: system prompt
        :param params: generation parameters
        :return: response
        """

        self.logger.info(f"Calling agent with prompt: {system[:50]}...")
        messages = self._get_messages(prompt, system)
        resp_format = {"type": "json_object"} if self.eval else {"type": "text"}

        # Retry with exponential backoff
        @retry(stop=stop_after_attempt(self._max_retries),
               wait=wait_exponential(multiplier=1, min=1, max=60),
               before=before_log(self.logger, logging.INFO),
               after=after_log(self.logger, logging.INFO))
        def call_api() -> str:
            try:
                self.logger.debug("Sending request to API")
                api_response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=params.max_completion_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    response_format=resp_format,
                )
                response = api_response.choices[0].message.content
                self.logger.info(f"Received response from API: {response[:50]}...")
                return response
            except Exception as e:
                if hasattr(e, 'status_code'):
                    if e.status_code == 400:
                        self.logger.error(f"Bad request error: {str(e)}")
                        raise e
                    elif e.status_code == 429:
                        self.logger.warning("Rate limit exceeded. Retrying with exponential backoff.")
                        raise e
                self.logger.error(f"Unexpected error occurred: {str(e)}")
                raise e

        # Call the API
        response = call_api()
        return response
