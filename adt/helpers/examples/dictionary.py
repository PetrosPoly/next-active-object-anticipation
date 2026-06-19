#TODO: Check if there is value with less than 2

dictionary = {
    'key A': 1.4,
    'key B': 3,
    'key C': 5
}

# Check if there is any value less than 2
value_less_than_2 = any(value < 2 for value in dictionary.values())

if value_less_than_2:
    print("There is at least one value less than 2.")
else:
    print("There are no values less than 2.")
