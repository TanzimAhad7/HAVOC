from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HAVOCModelLoader:
    """
    Utility class to load a causal LLM and its tokenizer.

    Example:
        llm = LLMWithTokenizer(
            model_name="meta-llama/Llama-Guard-3-8B",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model, tokenizer = llm.load()
    """

    def __init__(
        self,
        model_name: str,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer

        self.model = None
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
        )

        # Ensure pad token exists (important for batching)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self):
        """Load causal language model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(self.device_map)
        self.model.eval()
        return self.model

    def load(self):
        """Load both model and tokenizer."""
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer
