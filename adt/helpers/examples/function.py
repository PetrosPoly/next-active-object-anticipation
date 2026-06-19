def example_function():
    """
    This function returns multiple values.
    
    Returns:
    tuple: A tuple containing multiple values.
    """
    return "apple", "banana", "cherry"

# Assign the output of the function to variables
a, b, c = example_function()

# Print the variables in a vertical order
print(a)
print(b)
print(c)

# Assign the output of the function to variables
(a, 
 b, 
 c) = example_function()

# Print the variables in a vertical order
print()
print(a)
print(b)
print(c)