import numpy as np
import matplotlib.pyplot as plt
import random
import time

def linear_search(arr, target):
    for i, num in enumerate(arr):
        if num == target:
            return i
    return -1

def binary_search(arr, target):
    first = 0
    last = len(arr)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first+last)//2
        if arr[mid] == target:
            index = mid
        else:
            if target<arr[mid]:
                last = mid -1
            else:
                first = mid +1
    return index

def interpolation_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            if arr[left] == target:
                return left
            return -1

        estimated_pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        
        if arr[estimated_pos] == target:
            return estimated_pos
        elif arr[estimated_pos] < target:
            left = estimated_pos + 1
        else:
            right = estimated_pos - 1
    return -1

def jump_search(arr, target):
    length = len(arr)
    jump = int(math.sqrt(length))
    prev = 0

    while arr[min(jump, length)-1] < target:
        prev = jump
        jump += int(math.sqrt(length))
        if prev >= length:
            return -1

    for i in range(prev, min(jump, length)):
        if arr[i] == target:
            return i
    return -1

def fibonacci_search(arr, x):
    n = len(arr)
    fib_m2 = 0  
    fib_m1 = 1  
    fib_m = fib_m1 + fib_m2 

    while fib_m < n:
        fib_m2 = fib_m1
        fib_m1 = fib_m
        fib_m = fib_m1 + fib_m2

    offset = -1

    while fib_m > 1:
        i = min(offset + fib_m2, n - 1)
        if arr[i] < x:
            fib_m = fib_m1
            fib_m1 = fib_m2
            fib_m2 = fib_m - fib_m1
            offset = i
        elif arr[i] > x:
            fib_m = fib_m2
            fib_m1 -= fib_m1
            fib_m2 = fib_m - fib_m2
        else:
            return i 

    if fib_m1 and offset + 1 < n and arr[offset + 1] == x:
        return offset + 1
    return -1


numbers = list(range(1000000))
random.shuffle(numbers)
data = []
search_value = random.choice(numbers) 
sorted_numbers = sorted(numbers)

for i in range(16000, 1000001, 16000):
    current_array = numbers[:i]  
    sorted_array = sorted_numbers[:i]  

    start_time = time.perf_counter_ns()
    binary_search(sorted_array, search_value)
    sorted_time = time.perf_counter_ns() - start_time

    start_time = time.perf_counter_ns()
    binary_search(current_array, search_value)
    unsorted_time = time.perf_counter_ns() - start_time

    data.append((i, sorted_time, unsorted_time))

sizes, sorted_times, unsorted_times = zip(*data)

coeffs_sorted = np.polyfit(np.log(sizes), sorted_times, 1)
coeffs_unsorted = np.polyfit(np.log(sizes), unsorted_times, 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(sizes, sorted_times, s=10, label='Время (отсортированный)')
plt.plot(sizes, coeffs_sorted[0] * np.log(sizes) + coeffs_sorted[1], color='red', 
         label=f"Аппроксимация: {coeffs_sorted[0]:.2f}log(x) + {coeffs_sorted[1]:.2f}")
plt.xlabel("Количество элементов")
plt.ylabel("Время поиска (нс)")
plt.title("Зависимость времени поиска (отсортированный массив)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(sizes, unsorted_times, s=10, label='Время (неотсортированный)')
plt.plot(sizes, coeffs_unsorted[0] * np.log(sizes) + coeffs_unsorted[1], color='red', 
         label=f"Аппроксимация: {coeffs_unsorted[0]:.2f}log(x) + {coeffs_unsorted[1]:.2f}")
plt.xlabel("Количество элементов")
plt.ylabel("Время поиска (нс)")
plt.title("Зависимость времени поиска (неотсортированный массив)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
