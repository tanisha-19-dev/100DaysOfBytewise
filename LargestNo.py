
def find_largest_number():
    first_number = float(input("Enter first number: "))
    second_number = float(input("Enter second number: "))
    third_number = float(input("Enter third number: "))
    
    largest_number = max(first_number, second_number, third_number)
    
    print("The largest number is:", largest_number)

find_largest_number()
