#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point d'entree Hugging Face Spaces."""

try:
    from download_models import download_if_missing

    download_if_missing()
except Exception as exc:
    print(f"[WARN] {exc}")

from interface import demo


def main():
    demo.launch()


if __name__ == "__main__":
    main()
