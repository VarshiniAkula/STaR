# STaR
STaR: Self-Taught Reasoner on GSM8K (Llama-3.2-3B-Instruct)
Setup:
Base model: Llama-3.2-3B-Instruct (decoder-only)
Training dataset: GSM8K train split
Evaluation dataset: GSM8K test split
Compared methods:
• Zero-Shot Chain-of-Thought (CoT)
• Vanilla SFT (train on gold rationales from train set only)
• STaR (rationale generation + rationalization loop, then SFT)
The codebase contains three scripts, plus a shared config. It follows the same decoder-only SFT pattern but swaps in the required model and adds the STaR outer loop.
