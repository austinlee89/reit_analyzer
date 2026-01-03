from google import genai

# This python code is to see the list of current gemini model APIs available.

API_KEY = "AIzaSyBsUHzZMoLYzY39jaII-nPhEXtyFqx6JAs"
client = genai.Client(api_key=API_KEY)
models = client.models.list()

for model in models:
    print(model.name)
