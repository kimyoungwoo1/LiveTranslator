"""Generate a side-by-side markdown report from completed model benchmark outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def cell(value, max_chars=190):
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", " ")[:max_chars]


def read_model(name):
    path = RESULTS / (name + ".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "engine": name, "success": False, "error": "테스트를 실행하지 못했습니다"
    }


def render_report():
    quick, hy = read_model("quickmt"), read_model("hymt")
    models = {"quickmt": quick, "hymt": hy}
    text = [
        "# LiveTranslator offline en→ko model benchmark",
        "",
        "The input sentences and Korean references were independently authored to test business terminology, idioms, numbers and technical terms.",
        "**This is a CPU GitHub-hosted runner test, NOT the user's Core Ultra 5 225H Windows notebook.**",
        "All test data is English/Korean text. No workplace recording is uploaded, and the LiveTranslator EXE is unchanged.",
        "",
        "## Model comparison",
        "",
        "| Metric | QuickMT en-ko | Hy-MT2 1.8B Q4_K_M |",
        "|---|---:|---:|",
    ]
    for key, label, suffix in [
        ("chrF", "chrF (small sample, higher is better)", ""),
        ("p50_seconds", "Median translation per case", " s"),
        ("mean_seconds", "Mean translation per case", " s"),
        ("max_seconds", "Slowest translation per case", " s"),
        ("memory_mb_backend_after", "llama-server memory sample", " MiB"),
        ("memory_mb_process_after", "Benchmark Python memory sample", " MiB"),
    ]:
        def metric(model):
            value = model.get("metrics", {}).get(key)
            return str(value) + suffix if value is not None else "—"
        text.append(f"| {label} | {metric(quick)} | {metric(hy)} |")
    text += [
        "",
        "## Execution status",
        "",
    ]
    for name, model in models.items():
        if model.get("success"):
            text.append(f"- **{name}**: succeeded.")
        else:
            text.append(f"- **{name}**: failed/incomplete — {cell(model.get('error'), 500)}")
    text += [
        "",
        "## Clean English sentences (reference translation available)",
        "",
        "| Case | English input | Korean reference | QuickMT | Hy-MT2 |",
        "|---|---|---|---|---|",
    ]
    def indexed(result, group):
        return {row["id"]: row for row in result.get("cases", [])
                if row.get("group") == group}

    by_quick = indexed(quick, "clean")
    by_hy = indexed(hy, "clean")
    cases = json.loads((ROOT / "en_ko_cases.json").read_text(encoding="utf-8"))
    for entry in cases["clean"]:
        q = by_quick.get(entry["id"], {}).get("output")
        h = by_hy.get(entry["id"], {}).get("output")
        text.append(
            "| " + " | ".join(map(cell, [
                entry["id"], entry["en"], entry["ko"], q, h
            ])) + " |"
        )
    text += [
        "",
        "## Noisy-ASR inputs (qualitative only, not scored)",
        "",
        "| ID | Misheard English input | QuickMT | Hy-MT2 |",
        "|---|---|---|---|",
    ]
    for entry in cases["noisy"]:
        q = indexed(quick, "noisy").get(entry["id"], {}).get("output")
        h = indexed(hy, "noisy").get(entry["id"], {}).get("output")
        text.append(
            "| " + " | ".join(map(cell, [
                entry["id"], entry["en"], q, h
            ])) + " |"
        )
    text += [
        "",
        "## How to decide",
        "",
        "1. Verify **60억/60일/30일**, order terminology, proper nouns, and whether the sentence meaning is correct.",
        "2. Review important mistranslations manually: chrF measures character overlap, not trustworthiness.",
        "3. On this CPU runner, latency is per input text only and **does not include Whisper recognition time**.",
        "4. If Hy-MT2 is much slower, QuickMT may fit live subtitles better even if some sentences read less naturally.",
        "5. Repeat the chosen model on the actual Windows laptop before replacing Argos, because CPU speed and memory differ.",
        "",
        "**No winner is declared automatically**: real-time suitability requires both semantic quality and latency.",
        ""
    ]
    return "\n".join(text)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    rendered = render_report()
    (RESULTS / "report.md").write_text(rendered, encoding="utf-8")
    print(rendered)
