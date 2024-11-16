import requests
from dotenv import dotenv_values

from .base_model import BaseModel


class ChatGPTModel(BaseModel):

    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model="gpt-3.5-turbo"):
        BaseModel.__init__(self)
        openai_api_key = dotenv_values(".env")['CHATGPT_API_SECRET']
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

        # Check if the request was successful
        if response.status_code == 200:
            print("Response from OpenAI:", response.json())
            print('\n')
            print(response.json())
            print('\n')
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} {response.text}"
