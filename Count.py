def count_vowels():
    user_string = input("Enter a string: ")
    
    vowels = "aeiouAEIOU"
    
    vowel_count = 0
    
    for char in user_string:
        if char in vowels:
            vowel_count += 1
    
    print("The number of vowels is:", vowel_count)

count_vowels()
