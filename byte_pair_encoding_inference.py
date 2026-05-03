import sys

from byte_pair_encoding_common import chars_to_token_list, merge, prepareOccurrencesMap

INFERENCE_TEXT = "low roses"


def get_merge_criterion(token_map, merge_order):
    max_item = None
    min_rank = sys.maxsize
    for key in token_map.keys():
        if key in merge_order:
            if merge_order[key] < min_rank:
                min_rank = merge_order[key]
                max_item = key
    return max_item


def infer(encoding_text, merge_order, vocab, number_of_merge_iterations=10):
    token_list = chars_to_token_list(encoding_text)

    while number_of_merge_iterations >= 0:
        token_map, _ = prepareOccurrencesMap(token_list)
        item = get_merge_criterion(token_map, merge_order)
        if item is None:
            break
        token_list = merge(token_list, item)
        number_of_merge_iterations -= 1
    unknown = [t for t in token_list if t not in vocab]
    if unknown:
        raise ValueError(
            "Token(s) not in training vocabulary: " + str(sorted(set(unknown)))
        )
    text_encoding = [vocab[t] for t in token_list]
    return token_list, text_encoding


def run():
    from byte_pair_encoding_training import train, TRAINING_TEXT

    merge_order, vocab, _ = train(TRAINING_TEXT)
    token_list, text_encoding = infer(INFERENCE_TEXT, merge_order, vocab)
    print(f"Tokens: {token_list}")
    print(f"IDs: {text_encoding}")


if __name__ == "__main__":
    run()