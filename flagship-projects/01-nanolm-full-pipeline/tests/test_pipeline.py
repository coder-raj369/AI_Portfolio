from __future__ import annotations

import torch

from nanolm.data import build_instruction_pairs, build_pretrain_corpus, build_preference_pairs, train_tokenizer
from nanolm.model import TinyDecoderLM
from nanolm.train import dpo_loss, evaluate_model, grpo_loss, sft_finetune, train_base_model, train_grpo


def test_tokenizer_and_dataset_shape() -> None:
    corpus = build_pretrain_corpus()
    tokenizer = train_tokenizer(corpus, vocab_size=512)
    assert tokenizer.vocab_size > 256
    assert len(build_instruction_pairs()) >= 5
    assert len(build_preference_pairs()) >= 5


def test_model_forward_shapes() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=16, d_hidden=64)
    x = torch.tensor([tokenizer.encode("Paris is known for the Eiffel Tower.", allowed_special="all")[:10]], dtype=torch.long)
    out = model(x)
    assert out.shape == (1, 10, tokenizer.vocab_size)


def test_train_base_model_reduces_loss() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=24, n_layers=2, n_heads=4, max_seq_len=16, d_hidden=48)
    seqs = [tokenizer.encode(text, allowed_special="all")[:12] for text in ["Paris is the capital of France.", "Earth is a planet.", "Foxes are clever."]]
    result = train_base_model(model, seqs, epochs=3, batch_size=2, lr=5e-3)
    assert result.final_loss < result.mean_loss


def test_sft_finetune_runs() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=24, n_layers=1, n_heads=4, max_seq_len=16, d_hidden=48)
    pairs = [(tokenizer.encode(prompt, allowed_special="all"), tokenizer.encode(answer, allowed_special="all")) for prompt, answer in build_instruction_pairs()[:3]]
    result = sft_finetune(model, pairs, epochs=2, batch_size=2, lr=1e-3)
    assert result.final_loss > 0.0


def test_dpo_loss_is_finite() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=4, max_seq_len=12, d_hidden=32)
    ref = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=4, max_seq_len=12, d_hidden=32)
    chosen = torch.tensor(tokenizer.encode("The capital of France is Paris.", allowed_special="all")[:8], dtype=torch.long)
    rejected = torch.tensor(tokenizer.encode("The capital of France is Rome.", allowed_special="all")[:8], dtype=torch.long)
    loss = dpo_loss(model, ref, chosen, rejected)
    assert torch.isfinite(loss)


def test_evaluate_model_returns_score() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=4, max_seq_len=10, d_hidden=32)
    pairs = [(tokenizer.encode("What is 2 + 2?", allowed_special="all"), tokenizer.encode("4.", allowed_special="all"))]
    metrics = evaluate_model(model, pairs)
    assert "accuracy" in metrics


def test_grpo_loss_normalizes_group_rewards() -> None:
    rewards = torch.tensor([0.0, 1.0, 2.0])
    log_probs = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
    loss = grpo_loss(rewards, log_probs)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(log_probs.grad).all()


def test_train_grpo_runs_with_programmatic_reward() -> None:
    tokenizer = train_tokenizer(build_pretrain_corpus(), vocab_size=512)
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=4, max_seq_len=12, d_hidden=32)
    prompt = tokenizer.encode("Answer:", allowed_special="all")[:4]
    target = tokenizer.encode(" Paris.", allowed_special="all")[0]

    def reward_fn(_prompt: list[int], completion: list[int]) -> float:
        return float(completion[0] == target)

    before = model.token_embedding.weight.detach().clone()
    result = train_grpo(model, [prompt], reward_fn, epochs=2, group_size=3, max_new_tokens=2, lr=1e-3)
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert not torch.equal(before, model.token_embedding.weight.detach())
