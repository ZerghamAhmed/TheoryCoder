# query_client.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate
import openai
from groq import Groq
import os
from openai import OpenAI

class QueryClient:
    """
    A thin wrapper over LangChain-OpenAI, OpenAI’s Python SDK, or Groq,
    unified under a single .query(prompt) API.
    """
    def __init__(
        self,
        mode: str,
        model: str,
        temperature: float = 0.7,
        groq_model: str = None,
        groq_api_key_env: str = "GROQ_API_KEY",
    ):
        self.mode = mode
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

        if mode == "langchain_openai":
            # no persistent client needed; ChatOpenAI is lightweight
            pass

        elif mode == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # nothing else to init

        elif mode == "groq":
            key = os.getenv(groq_api_key_env)
            if not key:
                raise ValueError(f"Environment var {groq_api_key_env} is missing")
            self.groq = Groq(api_key=key)
            self.groq_model = groq_model or model

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def query(self, prompt: str) -> str:
        """Send `prompt` to the selected backend and return the generated text."""
        if self.mode == "langchain_openai":
            llm = ChatOpenAI(model_name=self.model, temperature=self.temperature)
            chat_prompt = HumanMessagePromptTemplate.from_template(prompt)
            return llm.invoke(chat_prompt.to_messages()).content

        elif self.mode == "openai":
            messages = [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                seed=42,
            )
            choice = resp.choices[0].message
            # If the SDK adds fingerprint metadata, you can pull it here
            return choice.content.strip(), resp

        elif self.mode == "groq":
            messages = [{"role": "user", "content": prompt}]
            resp = self.groq.chat.completions.create(
                model=self.groq_model,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()

        else:
            # should never happen
            raise RuntimeError(f"Unsupported mode: {self.mode}")
