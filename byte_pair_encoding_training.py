from byte_pair_encoding_common import convertTextToList, merge, prepareOccurrencesMap

def getMaxItem(token_map, token_index, merge_order, vocab):
    # Stop early when no merge candidates exist.
    if not token_map:
        return None

    # Track current best pair and its stats.
    max_occurrence = 0
    max_item = ""
    max_item_index = -1
    # Choose highest-frequency pair; break ties by earliest index.
    for key in token_map.keys():
        if token_map.get(key) > max_occurrence and token_map.get(key) > 2:
            max_occurrence = token_map.get(key)
            max_item_index = token_index[key]
            max_item = key
        elif token_map.get(key) == max_occurrence:
            if max_item_index > token_index[key]:
                max_occurrence = token_map.get(key)
                max_item_index = token_index[key]
                max_item = key

    # Record merge order and add merged token to vocabulary.
    if max_item:
        merge_order[max_item] = len(merge_order)
        vocab[max_item]= len(vocab)

    # Return the selected pair to be merged.
    return max_item


# Training text used for this BPE demo.
TRAINING_TEXT = "low lower lowest newest widest"


def train(training_text, number_of_merge_iterations=10):
    merge_order = {}
    vocab = {}
    token_list = convertTextToList(training_text, vocab)
    n = number_of_merge_iterations
    while n >= 0:
        token_map, token_index_map = prepareOccurrencesMap(token_list)
        item = getMaxItem(token_map, token_index_map, merge_order, vocab)
        if item is None:
            break
        token_list = merge(token_list, item)
        n -= 1
    return merge_order, vocab, token_list


def run():
    merge_order, vocab, token_list = train(TRAINING_TEXT)
    print(f"Built Vocabulary: {vocab}")
    print("\n")
    print(f"Merge Precedence: {merge_order}")
    print("\n")
    print(f"Last Round Token List: {token_list}")


if __name__ == "__main__":
    run()
