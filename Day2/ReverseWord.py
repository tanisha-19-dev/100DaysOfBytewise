def reverse_words(sentence):
    words = sentence.split()
    reversed_words = words[::-1]
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence

def main():
    sentence = input("Enter a sentence: ")
    reversed_sentence = reverse_words(sentence)
    print(reversed_sentence)

if __name__ == "__main__":
    main()
