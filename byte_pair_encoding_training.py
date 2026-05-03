def convertTextToList(text, vocab):
    # Start with a fresh token list for each run.
    token_list = []
    # Read input text one character at a time.
    for ch in text:
        # Add unseen base tokens to vocabulary.
        if ch not in token_list:
            vocab[str(ch)]= len(vocab)
        # Store each character as a token.
        token_list.append(str(ch))
    # Return initial character-level tokenization.
    return token_list


def merge(token_list, item):
    # Build a new list after applying one merge rule.
    newTokenList = []
    # Use index-based scan to skip merged pairs.
    index = 0
    # Continue until all tokens are processed.
    while index < len(token_list):
        # If adjacent tokens form the target pair, merge them.
        if "".join(token_list[index:index + 2]) == item:
            newTokenList.append(item)
            index = index + 2
        else:
            # Otherwise keep the current token unchanged.
            newTokenList.append(token_list[index])
            index = index + 1

    # Return the token list after one merge pass.
    return newTokenList


def prepareOccurrencesMap(tokenList):
    # Count how often each adjacent pair appears.
    token_map = {}
    # Keep first-seen index for tie-breaking later.
    token_index = {}
    # Iterate over every adjacent pair.
    for index in range(0, len(tokenList) - 1):
        item = tokenList[index] + tokenList[index + 1]
        # Create a tokenMap of all occurrences
        if item in token_map:
            token_map[item] = token_map[item] + 1
        else:
            token_map[item] = 1
            token_index[item] = index
    return token_map, token_index


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
text = "low lower lowest newest widest"

def run():
    # Reset state for rerunnable execution.
    mergeOrder = {}
    vocab = {}

    # Convert input text into initial token list.
    tokenList = convertTextToList(text, vocab)

    # Run a fixed number of merge iterations.
    number_of_merge_iterations = 10
    while number_of_merge_iterations >= 0:
        # Build pair frequency and first-index maps.
        token_map, token_index_map = prepareOccurrencesMap(tokenList)
        # Select best pair to merge in this round.
        item = getMaxItem(token_map, token_index_map, mergeOrder, vocab)
        # Stop if no valid pair remains.
        if item is None:
            break
        # Apply selected merge rule to token list.
        tokenList = merge(tokenList, item)
        # Move to next merge iteration.
        number_of_merge_iterations = number_of_merge_iterations - 1

    # Print final learned vocabulary and merge results.
    print(f"Built Vocabulary: {vocab}")
    print("\n")
    print(f"Merge Precedence: {mergeOrder}")
    print("\n")
    print(f"Last Round Token List: {tokenList}")


if __name__ == "__main__":
    run()
