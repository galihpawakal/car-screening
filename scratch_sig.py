import inspect
from google import genai
client = genai.Client(api_key="test")
sig = inspect.signature(client.interactions.create)
print("interactions.create signature:", sig)
