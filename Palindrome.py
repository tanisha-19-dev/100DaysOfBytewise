def check_palindrome():
    user_string = input("Enter a string: ")
    
    cleaned_string = user_string.replace(" ", "").lower()
    
    reversed_string = cleaned_string[::-1]
    
    if cleaned_string == reversed_string:
        print(user_string, "is a palindrome")
    else:
        print(user_string, "is not a palindrome")

check_palindrome()
