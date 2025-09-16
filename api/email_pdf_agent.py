import os
import base64
import json
import hashlib
import time
import typing as t
from email.message import EmailMessage
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

try:
    from PyPDF2 import PdfReader
    _USE_PYPDF2 = True
except Exception:
    _USE_PYPDF2 = False

USE_OPENAI = os.environ.get("USE_OPENAI", "true").lower() == "true"
OPENAI_MODEL = (
    os.environ.get("OPENAI_MODEL")
    or os.environ.get("OPENAI_VISION_MODEL")
    or "gpt-4o-mini"
)
try:
    from openai import OpenAI 
    _openai = OpenAI() if USE_OPENAI else None
except Exception:
    _openai = None
    USE_OPENAI = False

load_dotenv()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN", "")
GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "support.ai@diffrun.com")
GMAIL_CHAT_QUERY = os.environ.get(
    "GMAIL_CHAT_QUERY",
    'in:inbox has:attachment filename:pdf -from:me newer_than:7d '
    '-category:{promotions social} -label:"Agent/Chatted" -label:"Agent/Processing" -label:"Agent/PDFDone"'
)
CHAT_DONE_LABEL = os.environ.get("CHAT_DONE_LABEL", "Agent/Chatted")
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "60"))      
MAX_REPLY_CHARS = int(os.environ.get("MAX_REPLY_CHARS", "1800")) 
LOCK_LABEL = os.environ.get("LOCK_LABEL", "Agent/Processing")
DONE_LABEL = os.environ.get("DONE_LABEL", "Agent/PDFDone")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_access_token() -> str:
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": GOOGLE_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OAuth token error: {resp.status_code} {resp.text}")
    return resp.json()["access_token"]


def gmail_get(url_path: str, access_token: str, params: dict | None = None) -> dict:
    r = requests.get(
        f"https://gmail.googleapis.com/gmail/v1/users/me/{url_path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gmail GET {url_path}: {r.status_code} {r.text}")
    return r.json()


def gmail_post(url_path: str, access_token: str, payload: dict) -> dict:
    r = requests.post(
        f"https://gmail.googleapis.com/gmail/v1/users/me/{url_path}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Gmail POST {url_path}: {r.status_code} {r.text}")
    return r.json()


def gmail_modify_message_labels(message_id: str, add: list[str] | None, remove: list[str] | None, access_token: str) -> None:
    gmail_post(
        f"messages/{message_id}/modify",
        access_token,
        {"addLabelIds": add or [], "removeLabelIds": remove or []},
    )


def gmail_list_messages(access_token: str, query: str, max_results: int = 5) -> list[dict]:
    data = gmail_get("messages", access_token, params={"q": query, "maxResults": max_results})
    return data.get("messages", [])


def gmail_get_message(access_token: str, message_id: str) -> dict:
    return gmail_get(f"messages/{message_id}", access_token, params={"format": "full"})


def gmail_get_attachment(access_token: str, message_id: str, attachment_id: str) -> bytes:
    data = gmail_get(f"messages/{message_id}/attachments/{attachment_id}", access_token)
    return _b64url_decode(data["data"])


def gmail_ensure_label(access_token: str, name: str) -> str:
    labels = gmail_get("labels", access_token).get("labels", [])
    for lab in labels:
        if lab.get("name") == name:
            return lab["id"]
    created = gmail_post("labels", access_token, {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"})
    return created["id"]


def gmail_send_reply(access_token: str, thread_id: str, in_reply_to_msg_id: str, to_addr: str, subject: str, body: str) -> str:
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = GMAIL_SENDER
    msg["Subject"] = f"Re: {subject}" if not subject.lower().startswith("re:") else subject
    msg["In-Reply-To"] = in_reply_to_msg_id
    msg["References"] = in_reply_to_msg_id
    msg.set_content(body)

    raw = _b64url_encode(msg.as_bytes())

    res = gmail_post("messages/send", access_token, {"raw": raw, "threadId": thread_id})
    return res["id"]

def extract_pdf_text_per_page(pdf_bytes: bytes) -> list[str]:
    pages: list[str] = []

    if _USE_PYPDF2:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) > MAX_PAGES:
            raise ValueError(f"PDF too large: {len(reader.pages)} pages (limit {MAX_PAGES})")
        for i in range(len(reader.pages)):
            try:
                txt = (reader.pages[i].extract_text() or "").strip()
            except Exception:
                txt = ""
            pages.append(txt)
        return pages

    raise RuntimeError("No PDF backend available. Install pdfplumber or PyPDF2.")

def summarize_pages_with_openai(pages: list[str], filename: str) -> str:
    """Compact professional summary; cites page numbers for facts."""
    if not USE_OPENAI or _openai is None:
        # Minimal, local fallback if OpenAI is disabled
        joined = "\n".join([f"(p.{i+1}) {p[:400]}" for i, p in enumerate(pages) if p.strip()][:5])
        return f"Summary for {filename} (heuristic fallback):\n" + joined[:MAX_REPLY_CHARS]

    # Reduce prompt size: feed first 8 + middle 2 + last 2 non-empty pages (max)
    non_empty = [(i+1, p) for i, p in enumerate(pages) if p and p.strip()]
    if len(non_empty) > 12:
        head = non_empty[:8]
        tail = non_empty[-2:]
        mid = non_empty[len(non_empty)//2 - 1 : len(non_empty)//2 + 1]
        sample = head + mid + tail
    else:
        sample = non_empty

    # Compose content with page tags
    chunks = []
    for pg, txt in sample:
        chunks.append(f"[Page {pg}]\n{txt[:3000]}")  # guard per-page size
    doc_text = "\n\n".join(chunks)

    system = (
        "You analyze user-provided PDFs. Only use provided text. "
        "Never speculate. If a fact isn’t stated, say 'Not stated'. "
        "Cite page numbers in parentheses, e.g., (p.5). "
        "Be concise and executive-ready."
    )
    user = (
    f"Document: {filename}\n"
    "Write a short executive summary (≤150 words). Then:\n"
    "• 3–5 key points (with page cites)\n"
    "• 2 risks/assumptions (with page cites)\n"
    "• 2 action items\n\n"
    "Text:\n" + doc_text
)

    resp = _openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content or ""
    return out[:MAX_REPLY_CHARS]


def find_first_pdf_in_message(msg: dict) -> tuple[str, str] | None:
    def walk(parts: list[dict]) -> tuple[str, str] | None:
        for part in parts:
            mime = part.get("mimeType", "")
            filename = part.get("filename", "") or ""
            body = part.get("body", {}) or {}
            if mime == "application/pdf" or filename.lower().endswith(".pdf"):
                att_id = body.get("attachmentId")
                if att_id:
                    return filename or "document.pdf", att_id
            # nested multipart
            if "parts" in part:
                found = walk(part["parts"])
                if found:
                    return found
        return None

    payload = msg.get("payload", {}) or {}
    if payload.get("parts"):
        return walk(payload["parts"])
    # Single-part attachment rare; still check
    if (payload.get("mimeType") == "application/pdf") and payload.get("body", {}).get("attachmentId"):
        return (payload.get("filename") or "document.pdf", payload["body"]["attachmentId"])
    return None


def get_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def process_one_email(access_token: str) -> bool:
    msgs = gmail_list_messages(access_token, GMAIL_CHAT_QUERY, max_results=5)
    if not msgs:
        return False

    # Resolve label names (safe if globals not defined)
    lock_label_name = globals().get("LOCK_LABEL", os.environ.get("LOCK_LABEL", "Agent/Processing"))
    done_label_name = globals().get("DONE_LABEL", globals().get("CHAT_DONE_LABEL", os.environ.get("DONE_LABEL", os.environ.get("CHAT_DONE_LABEL", "Agent/PDFDone"))))

    for m in msgs:
        msg = gmail_get_message(access_token, m["id"])
        thread_id = msg.get("threadId")
        headers = msg.get("payload", {}).get("headers", []) or []
        subj = get_header(headers, "Subject") or "(no subject)"
        from_addr = get_header(headers, "From")
        to_addr_for_reply = (from_addr.split("<")[-1].rstrip(">") or "").strip() or from_addr

        # Ensure labels exist and check current state
        lock_label_id = gmail_ensure_label(access_token, lock_label_name)
        done_label_id = gmail_ensure_label(access_token, done_label_name)
        msg_label_ids = msg.get("labelIds") or []
        if lock_label_id in msg_label_ids or done_label_id in msg_label_ids:
            continue  # already being processed or done

        pdf_meta = find_first_pdf_in_message(msg)
        if not pdf_meta:
            continue  # skip this and try next match

        # Acquire lock BEFORE any heavy work to avoid races
        gmail_modify_message_labels(msg["id"], add=[lock_label_id], remove=None, access_token=access_token)

        try:
            filename, attachment_id = pdf_meta
            pdf_bytes = gmail_get_attachment(access_token, msg["id"], attachment_id)
            sha = sha256_hex(pdf_bytes)
            pages = extract_pdf_text_per_page(pdf_bytes)
            summary = summarize_pages_with_openai(pages, filename)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            body = (
                f"Here’s a concise analysis of '{filename}':\n\n"
                f"{summary}\n\n"
                f"—\nRef: {sha[:12]} • {stamp}"
            )

            message_id_hdr = get_header(headers, "Message-ID") or get_header(headers, "Message-Id") or ""
            sent_id = gmail_send_reply(
                access_token=access_token,
                thread_id=thread_id,
                in_reply_to_msg_id=message_id_hdr,
                to_addr=to_addr_for_reply,
                subject=subj,
                body=body,
            )

            gmail_modify_message_labels(
                msg["id"],
                add=[done_label_id],
                remove=[lock_label_id],
                access_token=access_token,
            )

            print(f"[OK] Replied {sent_id} on thread {thread_id} for {filename} ({len(pages)} pages)")
            return True 

        except Exception as e:
            try:
                gmail_modify_message_labels(msg["id"], add=None, remove=[lock_label_id], access_token=access_token)
            except Exception:
                pass
            print(f"[ERR] {type(e).__name__}: {e}")
            continue

    return False

def main() -> None:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REFRESH_TOKEN):
        raise SystemExit("Missing Google OAuth env vars. Set GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN.")

    if not (_USE_PYPDF2):
        raise SystemExit("Install at least one PDF library: `pip install pdfplumber` or `pip install PyPDF2`")

    print(f"Agent started. Query='{GMAIL_CHAT_QUERY}', done_label='{CHAT_DONE_LABEL}'")
    while True:
        try:
            token = get_access_token()
            handled = process_one_email(token)
            if not handled:
                time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("Stopping.")
            break
        except Exception as e:
            print(f"[ERR] {type(e).__name__}: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()