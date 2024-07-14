def list_of_squares():
    squares = [x ** 2 for x in range(1, 11)]
    return squares

if __name__ == "__main__":
    squares = list_of_squares()
    print(squares)
