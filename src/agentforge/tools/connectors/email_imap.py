"""IMAP email ingestion connector."""

from __future__ import annotations

import email
import imaplib
import os
from typing import Any

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, ToolResult, ToolError


class EmailIngestInput(BaseModel):
    subject: str | None = None
    sender: str | None = None
    limit: int = Field(default=5)


class EmailIngestTool(Tool):
    name = "email_imap_ingest"
    description = "Search IMAP inbox and fetch message snippets."
    input_schema = EmailIngestInput

    def run(self, data: BaseModel | dict[str, Any]) -> ToolResult:
        payload = EmailIngestInput.model_validate(data)
        host = os.getenv("IMAP_HOST")
        username = os.getenv("IMAP_USER")
        password = os.getenv("IMAP_PASSWORD")
        if not host or not username or not password:
            raise ToolError("IMAP credentials missing in environment variables.")
        with imaplib.IMAP4_SSL(host) as client:
            client.login(username, password)
            client.select("INBOX")
            criteria = []
            if payload.subject:
                criteria.append(f'SUBJECT "{payload.subject}"')
            if payload.sender:
                criteria.append(f'FROM "{payload.sender}"')
            search_query = " ".join(criteria) if criteria else "ALL"
            status, data_items = client.search(None, search_query)
            if status != "OK":
                raise ToolError("IMAP search failed")
            ids = data_items[0].split()[-payload.limit :]
            messages = []
            for msg_id in ids:
                status, msg_data = client.fetch(msg_id, "(RFC822)")
                if status != "OK":
                    continue
                msg = email.message_from_bytes(msg_data[0][1])
                body = _extract_body(msg)
                messages.append(
                    {
                        "subject": msg.get("Subject"),
                        "from": msg.get("From"),
                        "snippet": body[:500],
                    }
                )
        return ToolResult(output={"messages": messages})


def _extract_body(message: email.message.Message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True) or b""
                return payload.decode("utf-8", errors="replace")
        return ""
    payload = message.get_payload(decode=True) or b""
    return payload.decode("utf-8", errors="replace")
