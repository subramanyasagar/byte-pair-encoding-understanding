def chars_to_token_list(text):
    # Character-level tokens only; does not change vocabulary (for inference input).
    return [str(ch) for ch in text]


def convertTextToList(text, vocab):
    # Start with a fresh token list for each run.
    token_list = []
    # Read input text one character at a time.
    for ch in text:
        # Add unseen base tokens to vocabulary.
        if ch not in token_list:
            vocab[str(ch)] = len(vocab)
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
        if "".join(token_list[index : index + 2]) == item:
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
