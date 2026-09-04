#!/usr/bin/env python3
"""Create a standards-compliant exp1 .eml draft with the requested attachments."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "exp1"


def main() -> None:
    message = EmailMessage()
    message["To"] = "seong6466@gmail.com"
    message["Subject"] = "[NGSC-GRPO] EXP1 정리 완료 — internal/external 최종 결과"
    body = (REPORT / "EXP1_FINAL_REPORT_KO.md").read_text(encoding="utf-8")
    message.set_content(
        "안녕하세요.\n\nEXP1의 internal/external 최종 결과 정리를 완료했습니다. "
        "아래에는 전체 한국어 보고서를 본문으로 포함했으며, 모바일 확인용 결과 캡처와 원본 CSV를 첨부했습니다.\n\n"
        + body
    )
    attachments = (
        ("exp1_macro_summary_mobile.png", "image", "png"),
        ("exp1_final_results.csv", "text", "csv"),
        ("exp1_macro_summary.csv", "text", "csv"),
    )
    for filename, maintype, subtype in attachments:
        path = REPORT / filename
        if maintype == "text":
            message.add_attachment(
                path.read_text(encoding="utf-8"), subtype=subtype, filename=filename
            )
        else:
            message.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=filename)
    output = REPORT / "EXP1_COMPLETION_EMAIL.eml"
    output.write_bytes(message.as_bytes())
    print(output)


if __name__ == "__main__":
    main()
