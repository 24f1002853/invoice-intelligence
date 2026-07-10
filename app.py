from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def home():
    return {"status": "API Running"}


@app.post("/")
def extract(req: RequestBody):

    prompt = f"""
You are an expert information extraction system.

Extract information from the document.

Document:

{req.text}

The required JSON Schema is:

{json.dumps(req.schema, indent=2)}

Rules:

1. Return ONLY valid JSON.
2. Follow the JSON Schema EXACTLY.
3. Return exactly the required keys.
4. No extra keys.
5. Missing values must be null.
6. Dates must be YYYY-MM-DD.
7. Currency must be ISO4217 (USD, INR, EUR, GBP, JPY).
8. Numbers must be JSON numbers.
9. Booleans must be true/false.
10. Preserve line_items order.
11. Do NOT return markdown.
12. Do NOT explain anything.
"""

    try:

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        answer = response.text.strip()

        answer = answer.replace("```json", "")
        answer = answer.replace("```", "")
        answer = answer.strip()

        return json.loads(answer)

    except Exception:
        return {}