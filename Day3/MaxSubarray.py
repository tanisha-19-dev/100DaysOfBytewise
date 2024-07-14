def max_subarray(nums):
    max_sum = nums[0]
    current_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Example usage
print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6
