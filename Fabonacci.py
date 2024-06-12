def fibonacci_sequence():
    n_terms = int(input("How many terms? "))
    
    n1, n2 = 0, 1
    count = 0
    
    if n_terms <= 0:
        print("Please enter a positive integer")
    elif n_terms == 1:
        print("Fibonacci sequence up to", n_terms, "term:")
        print(n1)
    else:
        print("Fibonacci sequence:")
        while count < n_terms:
            print(n1, end=" ")
            nth = n1 + n2
            n1 = n2
            n2 = nth
            count += 1

fibonacci_sequence()
