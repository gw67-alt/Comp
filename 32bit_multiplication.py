def multiply_by_doubling(a, b):
    """
    Multiply two numbers using the Russian Peasant / Egyptian multiplication method.
    This method only uses doubling, halving, and addition operations.
    """
    # Initialize the result
    result = 0
    
    print(f"Multiplying {a} × {b} using doubling method:")
    print(f"{'Step':4} | {'Left':11} | {'Right':12} | {'Add to result?'}")
    print("-" * 4 + "-+-" + "-" * 11 + "-+-" + "-" * 12 + "-+-" + "-" * 15)
    
    step = 1
    while b > 0:
        # If b is odd, add the current value of a to the result
        if b % 2 == 1:
            result += a
            print(f"{step:4} | {a:11} | {b:12} | Yes, result = {result}")
        else:
            print(f"{step:4} | {a:11} | {b:12} | No")
        
        # Double a and halve b (integer division)
        a = a * 2  # This is the doubling step
        b = b // 2  # Integer division in Python
        
        step += 1
    
    return result

# Test the function with different examples
def main():
    examples = [
        (17, 23),
        (42, 13),
        (7, 8),
        (123, 2147483647)
    ]
    
    for a, b in examples:
        print("\nExample:")
        result = multiply_by_doubling(a, b)
        print(f"Final result: {result}")
        print(f"Verification: {a} × {b} = {a * b}")

if __name__ == "__main__":
    main()
