
def simple_calculator():
    
    first_number = float(input("Enter first number: "))
    
   
    second_number = float(input("Enter second number: "))
    
   
    operation = input("Enter operation (+, -, *, /): ")
    
   
    if operation == '+':
        result = first_number + second_number
    elif operation == '-':
        result = first_number - second_number
    elif operation == '*':
        result = first_number * second_number
    elif operation == '/':
        
        if second_number != 0:
            result = first_number / second_number
        else:
            print("Error: Division by zero is not allowed.")
            return
    else:
        print("Error: Invalid operation. Please enter one of +, -, *, /.")
        return
    

    print("The result is:", result)


simple_calculator()
