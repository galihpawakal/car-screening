from google import genai

client = genai.Client(api_key="test")
# Let's see what happens when calling client.interactions.create with fake key to inspect request payload
try:
    client.interactions.create(
        model="gemini-2.5-flash",
        input="Hello",
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        }
    )
except Exception as e:
    print("response_format call error (expected network/auth):", type(e), e)

try:
    client.interactions.create(
        model="gemini-2.5-flash",
        input="Hello",
        response_mime_type="application/json",
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
except Exception as e:
    print("response_mime_type call error:", type(e), e)
