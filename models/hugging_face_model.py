from transformers import pipeline

from .base_model import BaseModel


class HuggingFaceModel(BaseModel):

    model = None

    def __init__(self, model, task, device="cuda"):
        BaseModel.__init__(self)
        self.model = pipeline(model=model, task=task, device=device)

    def run(self, prompt):
        if self.model is None:
            raise ValueError("HuggingFaceModel is not initialized")
        answer = self.model(prompt)[0]
        return answer["generated_text"]
