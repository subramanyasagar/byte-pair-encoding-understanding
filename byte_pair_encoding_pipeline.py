"""
Run BPE training, then encode new text using the learned merge_order and vocab.
"""

from byte_pair_encoding_inference import INFERENCE_TEXT, infer
from byte_pair_encoding_training import TRAINING_TEXT, train


def main():
    merge_order, vocab, train_tokens = train(TRAINING_TEXT)
    print("--- Training ---")
    print(f"Built Vocabulary: {vocab}")
    print(f"Merge Precedence: {merge_order}")
    print(f"Training token list: {train_tokens}")
    print()
    print("--- Inference ---")
    token_list, ids = infer(INFERENCE_TEXT, merge_order, vocab)
    print(f"Input text: {INFERENCE_TEXT!r}")
    print(f"Tokens: {token_list}")
    print(f"IDs: {ids}")


if __name__ == "__main__":
    main()
