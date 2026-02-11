# scripts/chat.py
from src.config import load_config
from src.rag_pipeline import LocalRAG
from src.run_logger import finish_run, start_run


def main():
    cfg = load_config()

    rag = LocalRAG(cfg)
    rag.build_index()

    print(f"\n✅ RAG Chat Ready")
    print(f"LLM_PROVIDER={cfg.llm_provider}")
    print(f"OLLAMA_MODEL={cfg.ollama_model}")
    print(f"DATA_DIR={cfg.data_dir}")
    print("Type a question. Ctrl+C to exit.\n")

    while True:
        q = input("Human > ").strip()
        if not q:
            continue

        run = start_run(question=q, cfg=cfg)
        try:
            out = rag.query(q)
            finish_run(
                run=run,
                answer=out.get("answer", ""),
                sources=out.get("sources", []),
                timings=out.get("timings_ms", {}),
                context_used=out.get("context_used", ""),
                retrieval=out.get("retrieval", {}),
            )
        except Exception as exc:
            finish_run(
                run=run,
                answer="",
                sources=[],
                timings={},
                error=str(exc),
                context_used="",
                retrieval={},
            )
            raise

        print("\n" + out["answer"] + "\n")

        if out.get("sources"):
            print("Sources:")
            for s in out["sources"]:
                print(f"- {s.get('file')} (page {s.get('page')}), score={s.get('score'):.4f}")
            print("")


if __name__ == "__main__":
    main()
