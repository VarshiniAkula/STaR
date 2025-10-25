GSM8K_COT_SUFFIX = (
    "Let's think step by step, then give the final answer on a new line as "
    "#### <number>"
)

def build_zero_shot_prompt(question: str) -> str:
    return f"Q: {question}\nA: {GSM8K_COT_SUFFIX}\n"

def build_hint_prompt(question: str, gold_answer: str) -> str:
    return (
        f"Q: {question}\n"
        f"(Hint: The correct final answer is {gold_answer}. "
        f"Explain the steps that lead to it.)\nA: {GSM8K_COT_SUFFIX}\n"
    )
