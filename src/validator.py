from ray import serve
import re


@serve.deployment(ray_actor_options={"num_cpus": 0.5, "num_gpus": 0}, autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 5})
class Validator:
    def __init__(self):
        # Check incoming prompt for non-allowed characters. Return client error if non-allowed characters are present
        # Run inference
        self.regex = '^[a-zA-Z0-9 ?\',.-]+$'

    def validate(self, text: str) -> bool:
        if len(text) > 200:
            return False
        return bool(re.match(self.regex, text))
