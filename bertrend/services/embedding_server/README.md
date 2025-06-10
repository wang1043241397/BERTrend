# How to use the embedding server

## Launch the embedding server

- check the file `config/default_config.toml`

- run the embedding server

```
python start.py
```

## Check the embedding server
Ensure the API is launched.

```python
from bertrend.services.embedding_client import EmbeddingAPIClient

embedding_service_endpoint = "https://yourservice:1234"

# Initialize the API
api = EmbeddingAPIClient(embedding_service_endpoint)

# Get embedding model name
print(api.get_api_model_name())

# Transform text to embedding
text = "Hello, world!"
embedding = api.embed_query(text)
print(embedding)
```