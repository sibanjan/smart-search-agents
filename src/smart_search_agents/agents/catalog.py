"""The assistant choices offered in Setup and Settings, with what each costs the user.

The local models were benchmarked on 2026-11-07 (Intel Core Ultra 7 laptop, 32 GB, on battery):
five questions with known answers, twice each, through the real Ask route. Correct answers after
reading each one: Qwen3 4B 10/10 at a few minutes per answer, Llama 3.2 8/10 at under a minute,
Llama 3.1 8B 5/10. None made an image-category suggestion. Claude costs are rough estimates from the published per-token prices
for a typical question (about 20,000-60,000 input tokens across the agent's rounds).
"""

from __future__ import annotations

OLLAMA_DOWNLOAD = "https://ollama.com/download"

ASSISTANTS: list[dict] = [
    {
        "id": "off",
        "backend": "off",
        "title": "No assistant",
        "tagline": "Search and images only",
        "details": ["Nothing more to download", "Ask AI and category suggestions are hidden"],
        "privacy": "Everything stays on this computer.",
    },
    {
        "id": "ollama:llama3.2",
        "backend": "ollama",
        "model": "llama3.2",
        "title": "Llama 3.2 (3B)",
        "tagline": "On this computer · fast, usually right on simple lookups",
        "download_gb": 2.0,
        "details": [
            "2.0 GB download, runs on most laptops (8 GB RAM)",
            "Under a minute per question on our test laptop",
            "8 of 10 test questions right; sometimes answers from general knowledge",
            "Cannot suggest image categories",
        ],
        "privacy": "Everything stays on this computer.",
        "tested": True,
    },
    {
        "id": "ollama:qwen3:4b",
        "backend": "ollama",
        "model": "qwen3:4b",
        "title": "Qwen3 (4B)",
        "tagline": "On this computer · best local answers, but slow",
        "download_gb": 2.5,
        "details": [
            "2.5 GB download, runs on most laptops (8 GB RAM)",
            "10 of 10 test questions right, with sources",
            "Thinks before every step: 2-12 minutes per question on our test laptop",
            "Cannot suggest image categories",
        ],
        "privacy": "Everything stays on this computer.",
        "tested": True,
    },
    {
        "id": "ollama:llama3.1:8b",
        "backend": "ollama",
        "model": "llama3.1:8b",
        "title": "Llama 3.1 (8B)",
        "tagline": "On this computer · larger, but did worst in our test",
        "download_gb": 4.9,
        "details": [
            "4.9 GB download, 16 GB RAM recommended",
            "1-3 minutes per question on our test laptop",
            "5 of 10 test questions right: adds search filters nobody asked for",
            "Cannot suggest image categories",
        ],
        "privacy": "Everything stays on this computer.",
        "tested": True,
    },
    {
        "id": "claude:claude-opus-5-5",
        "backend": "claude",
        "model": "claude-opus-5-5",
        "title": "Claude Opus 5.5",
        "tagline": "Anthropic's cloud · best answers and image suggestions",
        "details": [
            "Fast; nothing to download",
            "Needs a Claude API key; you pay Anthropic per use",
            "Roughly $0.10-0.30 per question (estimate)",
        ],
        "privacy": "Your question and the file excerpts it reads are sent to Anthropic.",
        "recommended": True,
    },
    {
        "id": "claude:claude-sonnet-5-5",
        "backend": "claude",
        "model": "claude-sonnet-5-5",
        "title": "Claude Sonnet 5.5",
        "tagline": "Anthropic's cloud · very good answers at half the price of Opus",
        "details": [
            "Fast; nothing to download",
            "Needs a Claude API key; you pay Anthropic per use",
            "Roughly $0.05-0.15 per question (estimate)",
        ],
        "privacy": "Your question and the file excerpts it reads are sent to Anthropic.",
    },
    {
        "id": "claude:claude-haiku-5-5",
        "backend": "claude",
        "model": "claude-haiku-5-5",
        "title": "Claude Haiku 5.5",
        "tagline": "Anthropic's cloud · cheapest Claude, good for simple questions",
        "details": [
            "Fastest; nothing to download",
            "Needs a Claude API key; you pay Anthropic per use",
            "Usually under $0.01 per question (estimate)",
        ],
        "privacy": "Your question and the file excerpts it reads are sent to Anthropic.",
    },
]

PULLABLE = {a["model"] for a in ASSISTANTS if a["backend"] == "ollama"}


def has_model(installed: list[str], model: str) -> bool:
    """Ollama lists "llama3.2:latest" for a model pulled as "llama3.2"."""
    names = set(installed)
    return model in names or (":" not in model and f"{model}:latest" in names)
