from ray import serve
from transformers import AutoTokenizer, BatchEncoding


@serve.deployment(ray_actor_options={"num_cpus": 0.5, "num_gpus": 0}, autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 5})
class Tokenize:
    def __init__(self):
        checkpoint = "EleutherAI/gpt-j-6B"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def encode(self, text: str) -> BatchEncoding:
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, tokens: list):
        return self.tokenizer.decode(tokens)