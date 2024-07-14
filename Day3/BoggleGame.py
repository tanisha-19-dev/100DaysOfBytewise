def find_words(board, words):
    def dfs(board, word, i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], '#'
        found = any(dfs(board, word, i + di, j + dj, k + 1) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        board[i][j] = tmp
        return found

    found_words = set()
    for word in words:
        if any(dfs(board, word, i, j, 0) for i in range(len(board)) for j in range(len(board[0]))):
            found_words.add(word)
    return found_words

# Example usage
board = [
    ['o', 'a', 'a', 'n'],
    ['e', 't', 'a', 'e'],
    ['i', 'h', 'k', 'r'],
    ['i', 'f', 'l', 'v']
]
words = ["oath", "pea", "eat", "rain"]
print(find_words(board, words))  # Output: {'eat', 'oath'}
