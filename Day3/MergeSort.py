def merge_sorted_arrays(arr1, arr2):
    merged_array = []
    i, j = 0, 0
    
    while i < arr1.length and j < arr2.length:
        if arr1[i] < arr2[j]:
            merged_array.append(arr1[i])
            i += 1
        else:
            merged_array.append(arr2[j])
            j += 1
    
    while i < arr1.length:
        merged_array.append(arr1[i])
        i += 1
    
    while j < arr2.length:
        merged_array.append(arr2[j])
        j += 1
    
    return merged_array

arr1 = [1, 3, 5]
arr2 = [2, 4, 6]
result = merge_sorted_arrays(arr1, arr2)
print(result)  # Output: [1, 2, 3, 4, 5, 6]
