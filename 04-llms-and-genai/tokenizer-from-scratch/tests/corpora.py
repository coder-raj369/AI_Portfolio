"""Training corpus shared by the test modules.

Deliberately mixes English prose, code, numbers and non-Latin scripts so the
merge table has to cope with more than one register.
"""

CORPUS = (
    "The quick brown fox jumps over the lazy dog. " * 40
    + "Machine learning models learn representations from data. " * 40
    + "def train(model, data):\n    return model.fit(data)\n" * 20
    + "Numbers like 2026, 42 and 3.14159 appear in text. " * 20
    + "Unicode: cafe, naive, 日本語, emoji 🚀 and math ∑∫∂. " * 20
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
