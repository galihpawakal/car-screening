from google import genai
client = genai.Client(api_key="test")
print("client attributes:", [a for a in dir(client) if not a.startswith("_")])
if hasattr(client, "interactions"):
    print("interactions attributes:", [a for a in dir(client.interactions) if not a.startswith("_")])
if hasattr(client, "models"):
    print("models attributes:", [a for a in dir(client.models) if not a.startswith("_")])
