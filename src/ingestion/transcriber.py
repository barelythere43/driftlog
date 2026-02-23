"""Transcribe handwritten journal images via AWS Textract + Claude vision."""
import asyncio
import base64
import json
import logging
import re

import boto3

from src.generation.generator import _get_client

logger = logging.getLogger(__name__)

_textract_client = boto3.client("textract", region_name="us-east-1")

TRANSCRIPTION_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 8192

SYSTEM_PROMPT = (
    "You are a handwriting transcription and metadata extraction assistant. "
    "Respond with a JSON array only: one object per journal entry, each with transcription and metadata."
)

FALLBACK_ENTRY = {
    "transcription": "",
    "metadata": {"date": None, "location": None, "country": None, "tags": []},
}


def _extract_text_textract(image_bytes: bytes) -> str:
    """Call Textract detect_document_text; return LINE blocks joined by newlines."""
    response = _textract_client.detect_document_text(Document={"Bytes": image_bytes})
    lines = []
    for block in response["Blocks"]:
        if block.get("BlockType") == "LINE" and "Text" in block:
            lines.append(block["Text"])
    return "\n".join(lines)


async def transcribe_journal_images(images: list[dict]) -> list[dict]:
    """Run Textract on each image, then Claude vision; return a list of entries, each with transcription and metadata."""
    if not images:
        return []

    # Step 1: Textract per image (async thread), combine text
    textract_parts: list[str] = []
    for img in images:
        try:
            image_bytes = base64.b64decode(img["data"])
        except Exception as e:
            logger.warning("Failed to base64-decode image for Textract: %s", e)
            continue
        try:
            text = await asyncio.to_thread(_extract_text_textract, image_bytes)
            if text.strip():
                textract_parts.append(text)
        except Exception as e:
            logger.warning("Textract failed on image, continuing without its OCR: %s", e)

    textract_text = "\n\n".join(textract_parts) if textract_parts else ""

    # Step 2 & 3: Build content: text block + all image blocks, call Claude
    user_text = f"""You are reading handwritten journal pages. The handwriting is often quick, mixed cursive and print, with abbreviations, possible smudges, angles, or crossed-out parts.

Here is raw OCR text from Amazon Textract (may be noisy due to handwriting):
{textract_text}

You can also SEE the actual images of the pages.

Tasks:
1. Look for date headers or clear entry boundaries (new dates, horizontal lines, "Dear diary", blank lines between sections, etc.). If a page contains MULTIPLE dated or distinct entries, split them into separate items. A page with a single entry yields one item; multiple entries yield multiple items.
2. For each entry, use BOTH the images and the OCR text to produce an accurate transcription. Fix OCR errors by looking at letter shapes, fix spelling/context (e.g., "pho" not "rho", place names, currency amounts).
3. For each entry, extract metadata: date of entry, location (the city/place where the author physically was when writing — NOT places mentioned for comparison), country, and relevant tags from: food, coffee, coworking, accommodation, transport, nightlife, culture, nature, fitness, shopping

Respond with a JSON array only, no other text. One object per entry. A single entry is an array of length 1.
[{{"transcription": "...", "metadata": {{"date": null, "location": null, "country": null, "tags": []}}}}, ...]
Use null for any metadata field you cannot confidently determine."""

    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/jpeg"),
                    "data": img["data"],
                },
            }
        )

    client = _get_client()
    try:
        response = await client.messages.create(
            model=TRANSCRIPTION_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        logger.exception("Claude vision transcription failed: %s", e)
        raise

    # Step 4: Parse response — strip markdown fences, then expect a list of {transcription, metadata}
    raw = response.content[0].text if response.content else ""
    if not raw:
        return []

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Transcription response was not valid JSON, using raw text as single entry")
        return [{"transcription": raw, "metadata": FALLBACK_ENTRY["metadata"]}]

    if isinstance(out, list):
        entries = []
        for item in out:
            if isinstance(item, dict) and "transcription" in item and "metadata" in item:
                entries.append({"transcription": item["transcription"], "metadata": item["metadata"]})
            else:
                entries.append({**FALLBACK_ENTRY})
        return entries if entries else [FALLBACK_ENTRY]
    if isinstance(out, dict) and "transcription" in out and "metadata" in out:
        return [out]
    return [FALLBACK_ENTRY]
