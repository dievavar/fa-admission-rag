#!/usr/bin/env python3
"""Convert scanned PDF to searchable PDF using installed Tesseract (no OCRmyPDF).

Requires: tesseract executable, pypdfium2, Pillow, pypdf.
Usage:
  python ocr_tesseract_pdf.py data/raw/rules_bachelor_2026.pdf \
      data/processed/ocr/rules_bachelor_2026_searchable.pdf
Optional:
  python ocr_tesseract_pdf.py input.pdf output.pdf --tessdata-dir data/tessdata
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Сканированный PDF -> PDF с текстовым слоем (Tesseract)")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--tessdata-dir", type=Path, default=None)
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        parser.error(f"Исходный PDF не найден: {inp}")
    if inp == out:
        parser.error("Нельзя перезаписывать оригинальный PDF")
    if out.exists():
        parser.error(f"Выходной файл уже существует: {out}")
    if args.dpi < 100 or args.dpi > 400:
        parser.error("Рекомендуемый диапазон разрешения: 100-400 DPI")
    if not shutil.which("tesseract"):
        parser.error("Команда tesseract не найдена в PATH")

    tessdata_dir = args.tessdata_dir
    if tessdata_dir is not None:
        tessdata_dir = tessdata_dir.resolve()
        if not tessdata_dir.is_dir():
            parser.error(f"Папка языковых данных не найдена: {tessdata_dir}")

    test_cmd = ["tesseract", "--list-langs"]
    if tessdata_dir is not None:
        test_cmd += ["--tessdata-dir", str(tessdata_dir)]
    lang_list = subprocess.run(test_cmd, capture_output=True, text=True, check=True).stdout
    langs = set(lang_list.splitlines())
    missing = set(args.lang.split("+")) - langs
    if missing:
        parser.error(f"Отсутствуют языки Tesseract: {', '.join(sorted(missing))}. Проверь --tessdata-dir")

    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(inp))
    writer = PdfWriter()
    total = len(pdf)
    errors: list[int] = []
    try:
        with tempfile.TemporaryDirectory(prefix="fa-tess-ocr-") as tmp:
            tmpdir = Path(tmp)
            for page_idx in range(total):
                page = pdf[page_idx]
                bitmap = page.render(scale=args.dpi / 72)
                picture = bitmap.to_pil()
                png = tmpdir / "page.png"
                pdf_prefix = tmpdir / "page_ocr"
                picture.save(png)
                picture.close()
                bitmap.close()
                page.close()

                cmd = ["tesseract", str(png), str(pdf_prefix)]
                if tessdata_dir is not None:
                    cmd += ["--tessdata-dir", str(tessdata_dir)]

                cmd += [
                    "-l", args.lang,
                    "--oem", "1",
                    "-c", "tessedit_create_pdf=1",
                    "-c", "tessedit_create_txt=0",
                ]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"Ошибка Tesseract на странице {page_idx + 1}: {proc.stderr[-1000:]}")

                recognized = PdfReader(str(pdf_prefix) + ".pdf")
                page_pdf = recognized.pages[0]
                extracted = page_pdf.extract_text() or ""
                meaningful = sum(ch.isalnum() for ch in extracted)
                print(f"Страница {page_idx + 1}/{total}: {meaningful} букв и цифр", flush=True)
                if meaningful < 20:
                    errors.append(page_idx + 1)
                writer.add_page(page_pdf)
                png.unlink(missing_ok=True)
                (tmpdir / "page_ocr.pdf").unlink(missing_ok=True)

        if errors:
            raise RuntimeError(
                "Похоже, на некоторых страницах нет распознанного текста: "
                + ", ".join(map(str, errors))
                + ". Результат не сохранён — проверь оригинал."
            )
        with out.open("wb") as f:
            writer.write(f)
        print(f"Готово: {out}\nСтраниц: {total}. Дальше проверь текст важных пунктов и таблиц.")
    finally:
        writer.close()
        pdf.close()


if __name__ == "__main__":
    main()
