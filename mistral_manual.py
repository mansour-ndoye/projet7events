import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("❌ Clé Mistral introuvable")
    exit()

client = Mistral(api_key=api_key)

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": "Réponds simplement : Bonjour Puls-Events !"
        }
    ],
)

print("✅ Connexion à Mistral réussie")
print("Réponse :", response.choices[0].message.content)