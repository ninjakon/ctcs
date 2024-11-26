import requests

from .base_model import BaseModel


class ChatGPTModel(BaseModel):

    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model="gpt-3.5-turbo", openai_api_key=None):
        if openai_api_key is None:
            raise ValueError("OpenAI API key must be provided")
        BaseModel.__init__(self)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        self.model = model

    def run(self, prompt):
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        response = requests.post(self.url, headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} {response.text}"
