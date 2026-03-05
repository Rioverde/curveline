"""
dataset.py — Load TruthfulQA from HuggingFace and format for curvature experiment.

Each example is a (text, is_hallucination) tuple where:
- is_hallucination=False means the text is factually correct
- is_hallucination=True means the text contains a hallucination/false claim
"""

from datasets import load_dataset as hf_load_dataset


def load_truthfulqa():
    """
    Load TruthfulQA dataset and return list of (text, is_hallucination, question) tuples.

    For each question:
    - FACTUAL: "{question} {best_answer}" with is_hallucination=False
    - HALLUCINATION: "{question} {incorrect_answers[0]}" with is_hallucination=True

    The third element (question) allows computing curvature on answer tokens only.
    Returns ~817 factual + ~817 hallucination = ~1634 total examples.
    """
    dataset = hf_load_dataset("truthful_qa", "generation", split="validation")

    examples = []
    for row in dataset:
        question = row["question"].strip()
        best_answer = row["best_answer"].strip()
        incorrect_answers = row["incorrect_answers"]

        # Factual example
        factual_text = f"{question} {best_answer}"
        examples.append((factual_text, False, question))

        # Hallucination example (use first incorrect answer if available)
        if incorrect_answers:
            hallucination_text = f"{question} {incorrect_answers[0].strip()}"
            examples.append((hallucination_text, True, question))

    return examples


# For backward compatibility with existing code
try:
    from exp_hallucination.sentences import SENTENCES as MANUAL_SENTENCES
except ImportError:
    try:
        from sentences import SENTENCES as MANUAL_SENTENCES
    except ImportError:
        MANUAL_SENTENCES = []

# Default SENTENCES variable — uses TruthfulQA if no manual sentences available
if MANUAL_SENTENCES:
    SENTENCES = MANUAL_SENTENCES
else:
    SENTENCES = load_truthfulqa()


def load_dataset_combined(use_manual=True, use_truthfulqa=True):
    """
    Load and combine dataset sources.

    Args:
        use_manual: Include manually curated sentences from sentences.py
        use_truthfulqa: Include TruthfulQA examples

    Returns:
        List of (text, is_hallucination) tuples
    """
    combined = []

    if use_manual and MANUAL_SENTENCES:
        # Manual sentences are 2-tuples (text, is_hallucination) — no question prefix
        for item in MANUAL_SENTENCES:
            if isinstance(item, tuple) and len(item) == 2:
                combined.append((item[0], item[1], ""))
            elif isinstance(item, tuple) and len(item) == 3:
                combined.append(item)
            else:
                combined.append((item, False, ""))

    if use_truthfulqa:
        combined.extend(load_truthfulqa())

    return combined


if __name__ == "__main__":
    print("Loading TruthfulQA dataset...")
    examples = load_truthfulqa()

    total = len(examples)
    factual = sum(1 for _, is_h, *_ in examples if not is_h)
    hallucinations = sum(1 for _, is_h, *_ in examples if is_h)
    avg_len = sum(len(text) for text, *_ in examples) / total if total > 0 else 0

    print(f"\n=== Dataset Stats ===")
    print(f"Total examples:      {total}")
    print(f"Factual examples:    {factual}")
    print(f"Hallucination exs:   {hallucinations}")
    print(f"Avg text length:     {avg_len:.1f} chars")

    print(f"\n=== Sample Entries ===")
    for i, (text, is_h, *_) in enumerate(examples[:6]):
        label = "HALLUCINATION" if is_h else "FACTUAL      "
        preview = text[:120] + ("..." if len(text) > 120 else "")
        print(f"[{i+1}] {label} | {preview}")

    if MANUAL_SENTENCES:
        print(f"\nManual sentences available: {len(MANUAL_SENTENCES)}")
    else:
        print(f"\nNo manual sentences found (sentences.py not present).")
