"""
Renders the RAG graph (rag/graph.py) as a PNG flowchart.

Usage:
    uv run python scripts/visualize_graph.py

Output:
    graph.png in the current working directory.

Notes:
    - Builds the graph WITHOUT a checkpointer (no Postgres connection
      needed) since visualization only needs the node/edge structure,
      not a runnable instance.
    - Rendering uses LangGraph's default Mermaid API renderer
      (mermaid.ink), which requires normal internet access. If you're
      offline or want to avoid the external API, see the
      draw_mermaid()-only fallback at the bottom of this script.
"""

import os
import sys

from nxb_chatbot.rag.graph import _build_graph

OUTPUT_DIR = os.path.join("data", "graph")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "graph.png")
FALLBACK_PATH = os.path.join(OUTPUT_DIR, "graph.mmd")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    builder = _build_graph()
    graph = builder.compile()

    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(png_bytes)

    print(f"Saved graph flowchart to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to render graph via the Mermaid API: {exc}")
        print("Falling back to raw Mermaid text (graph.mmd) instead.")

        from nxb_chatbot.rag.graph import _build_graph as _bg

        builder = _bg()
        graph = builder.compile()
        mermaid_text = graph.get_graph(xray=True).draw_mermaid()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(FALLBACK_PATH, "w", encoding="utf-8") as f:
            f.write(mermaid_text)

        print(
            f"Saved {FALLBACK_PATH} — paste its contents into "
            "https://mermaid.live to view/export as an image."
        )
        sys.exit(1)