from collections import deque

# Create a deque and populate it with some sample data
streaks = {
    "obj_1": deque([(1, "event1"), (2, "event2"), (3, "event3"), (4, "event4")])
}

# Define a cutoff time
cutoff_time_ns = 3

# Print the original deque
print("Original deque:", streaks["obj_1"])

# Loop to remove elements from the left of the deque while conditions are met
while streaks["obj_1"] and streaks["obj_1"][0][0] < cutoff_time_ns:
    streaks["obj_1"].popleft()

# Print the deque after removing elements
print("Deque after popleft operations:", streaks["obj_1"])
