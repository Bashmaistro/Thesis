import json

class Config:
    def __init__(self, config_path="conf.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
    
    def get(self, *keys, default=None):
        """Fetch nested keys safely, return default if not found."""
        data = self.config
        for key in keys:
            data = data.get(key, default) if isinstance(data, dict) else default
        return data