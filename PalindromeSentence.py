import string

def is_palindrome_sentence(sentence):
    cleaned_sentence = ''.join(char.lower() for char in sentence if char.isalnum())
    return cleaned_sentence == cleaned_sentence[::-1]

def main():
    sentence = input("Enter a sentence: ")
    if is_palindrome_sentence(sentence):
        print(f'"{sentence}" is a palindrome.')
    else:
        print(f'"{sentence}" is not a palindrome.')

if __name__ == "__main__":
    main()
