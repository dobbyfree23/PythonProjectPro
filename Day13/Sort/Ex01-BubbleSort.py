'''
버블정렬(Bubble sort)
    인접한 두 원소를 비교하여 정렬하는 알고리즘 중 하나로,
    가장 간단한 정렬 알고리즘 중 하나입니다.

    버블정렬의 시간복잡도는 최악의 경우 O(n^2)
'''

def bubble_sort(arr):
    n = len(arr)  # 배열의 길이를 구합니다.
    for i in range(n):  # 0부터 n-1까지 반복합니다.
        for j in range(n - i - 1):  # 0부터 n-i-2까지 반복합니다.
            if arr[j] > arr[j+1]:  # 인접한 두 원소를 비교합니다.
                # 만약 앞의 원소가 뒤의 원소보다 크면, 두 원소의 위치를 바꿉니다.
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr  # 정렬된 배열을 반환합니다.


'''
import matplotlib.pyplot as plt
import numpy as np

def plot_bars(arr):
    n = len(arr)
    bars = plt.bar(range(n), arr)
    plt.xticks(range(n), arr)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, str(int(height)), ha='center', va='bottom')

def animate_bubble_sort(arr):
    fig, ax = plt.subplots()
    plot_bars(arr)
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                ax.clear()
                plot_bars(arr)
                plt.pause(0.1)
    plt.show()

arr = np.random.randint(1, 50, 20)
animate_bubble_sort(arr)
'''