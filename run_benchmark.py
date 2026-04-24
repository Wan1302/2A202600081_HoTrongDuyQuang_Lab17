import argparse
from pathlib import Path

from lab17.benchmark import run_benchmark
from lab17.llm import create_llm_client
from lab17.reporting import write_benchmark_markdown, write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lab 17 benchmark.")
    parser.add_argument(
        "--llm",
        default="auto",
        help="LLM provider: auto, none, or openai. auto uses OpenAI when OPENAI_API_KEY exists.",
    )
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios.")
    args = parser.parse_args()

    llm = create_llm_client(args.llm, args.model)
    summary = run_benchmark(
        output_dir=Path("reports"),
        data_root=Path("data"),
        llm=llm,
        limit=args.limit,
    )
    write_markdown_report(summary, Path("reports/report.md"))
    write_benchmark_markdown(summary, Path("BENCHMARK.md"))
    print("Generated reports/report.json, reports/report.md, and BENCHMARK.md")


if __name__ == "__main__":
    main()
