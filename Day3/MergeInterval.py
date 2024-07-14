def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort the intervals by the start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged

# Example usage
intervals = [(1, 3), (2, 6), (8, 10), (15, 18)]
print(merge_intervals(intervals))  # Output: [(1, 6), (8, 10), (15, 18)]

intervals = [(1, 4), (4, 5)]
print(merge_intervals(intervals))  # Output: [(1, 5)]
