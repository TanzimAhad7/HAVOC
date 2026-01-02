from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HAVOCModelLoader:
    def __init__(
        self,
        model_name: str,
        torch_dtype=torch.float16,
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = False,
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer

        self.model = None
        self.tokenizer = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.use_fast_tokenizer,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",              # ✅ correct
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        return self.model

    def load(self):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer
