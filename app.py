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


@app.post("/extract")
def extract(req: RequestBody):

    prompt = f"""
You are an expert invoice extraction assistant.

Extract information from the invoice.

Invoice Text:

{req.text}

Return ONLY a valid JSON object.

The JSON MUST exactly follow this schema:

{json.dumps(req.schema, indent=2)}

Rules:

- Return ONLY JSON.
- Do NOT return markdown.
- Do NOT explain anything.
- Return EXACTLY the schema keys.
- No extra keys.
- Missing values -> null.
- vendor = biller's proper name exactly as written.
- currency = ISO 4217 code only (USD, EUR, GBP, INR, JPY).
- total_amount = integer.
- invoice_date = YYYY-MM-DD.
- due_in_days = integer.
- is_paid = boolean.
- priority MUST be one of:
  low
  normal
  high
  urgent
- contact_email MUST be lowercase.
- line_items MUST always be an array.
- Preserve the order of line_items.
- unit_price MUST be integer.
- quantity MUST be integer.
- item_count MUST equal length(line_items).
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

        data = json.loads(answer)

        # Normalize priority
        if isinstance(data.get("priority"), str):
            data["priority"] = data["priority"].lower()

        # Normalize email
        if isinstance(data.get("contact_email"), str):
            data["contact_email"] = data["contact_email"].lower()

        # Ensure line_items exists
        if data.get("line_items") is None:
            data["line_items"] = []

        # Compute item_count
        data["item_count"] = len(data["line_items"])

        return data

    except Exception as e:

        return {
            "error": str(e)
        }
