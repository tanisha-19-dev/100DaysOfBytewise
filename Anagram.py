def are_anagrams(str1, str2):
    cleaned_str1 = sorted(str1.replace(" ", "").lower())
    cleaned_str2 = sorted(str2.replace(" ", "").lower())
    return cleaned_str1 == cleaned_str2

def main():
    str1 = input("Enter the first string: ")
    str2 = input("Enter the second string: ")
    if are_anagrams(str1, str2):
        print(f'"{str1}" and "{str2}" are anagrams.')
    else:
        print(f'"{str1}" and "{str2}" are not anagrams.')

if __name__ == "__main__":
    main()
