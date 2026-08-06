from typing import Dict, List, Any
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch

class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("cuda")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        """
        # process input
        inputs = data.pop("inputs", data)
        voice_description = data.pop("voice_description", "data")
        parameters = data.pop("parameters", None)

        gen_kwargs = {"min_new_tokens": 10}
        if parameters is not None:
            gen_kwargs.update(parameters)

        # preprocess
        inputs = self.tokenizer(
            text=[inputs],
            padding=True,
            return_tensors="pt",).to("cuda")
        voice_description = self.tokenizer(
            text=[voice_description],
            padding=True,
            return_tensors="pt",).to("cuda")

        # pass inputs with all kwargs in data
        with torch.autocast("cuda"):
            outputs = self.model.generate(**voice_description, prompt_input_ids=inputs.input_ids, **gen_kwargs)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy().tolist()

        return [{"generated_audio": prediction}]