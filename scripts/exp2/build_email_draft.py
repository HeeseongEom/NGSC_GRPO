#!/usr/bin/env python3
"""Create a ready-to-send EXP2 MIME email without requiring mail credentials."""

from __future__ import annotations

import mimetypes
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "exp2"


def main() -> None:
    body_path = REPORT / "EXP2_FINAL_REPORT_KO.md"
    attachments = (
        REPORT / "exp2_macro_summary_mobile.png",
        REPORT / "exp2_macro_summary.csv",
        REPORT / "exp2_final_results.csv",
    )
    if not body_path.is_file():
        raise FileNotFoundError(body_path)
    for path in attachments:
        if not path.is_file():
            raise FileNotFoundError(path)

    message = EmailMessage(policy=SMTP)
    message["To"] = "seong6466@gmail.com"
    message["Subject"] = "[NGSC-GRPO] EXP2 8단계 완료"
    message.set_content(body_path.read_text(encoding="utf-8"), charset="utf-8")
    for path in attachments:
        mime, _ = mimetypes.guess_type(path.name)
        main_type, sub_type = (mime or "application/octet-stream").split("/", 1)
        message.add_attachment(path.read_bytes(), maintype=main_type, subtype=sub_type, filename=path.name)

    output = REPORT / "EXP2_COMPLETION_EMAIL.eml"
    output.write_bytes(message.as_bytes())
    print(output)


if __name__ == "__main__":
    main()
