'''
삽입 정렬(Insertion Sort)
    리스트의 모든 요소를 앞에서부터 차례대로 이미 정렬된 부분과 비교하여 자신의 위치를 찾아 삽입하는 알고리즘.
    O(n^2)의 시간복잡도를 갖는다.

def insertion_sort(arr):
    n = len(arr)  # 배열의 길이를 구합니다.
    for i in range(1, n):  # 1부터 n-1까지 반복합니다.
        key = arr[i]  # 현재 위치의 값을 key 변수에 저장합니다.
        j = i - 1  # 현재 위치의 이전 위치를 j 변수에 저장합니다.
        while j >= 0 and arr[j] > key:  # 이전 위치부터 첫 번째 위치까지 반복합니다.
            # 현재 위치의 값을 이전 위치로 이동합니다.
            arr[j+1] = arr[j]
            j -= 1  # 이전 위치로 이동합니다.
        arr[j+1] = key  # key 값을 현재 위치에 저장합니다.
    return arr  # 정렬된 배열을 반환합니다.
'''
def insertion_sort(arr):
    n = len(arr)  # 배열의 길이를 구합니다.
    for i in range(1, n):  # 1부터 n-1까지 반복합니다.
        key = arr[i]  # 현재 위치의 값을 key 변수에 저장합니다.
        j = i - 1  # 현재 위치의 이전 위치를 j 변수에 저장합니다.
        while j >= 0 and arr[j] > key:  # 이전 위치부터 첫 번째 위치까지 반복합니다.
            # 현재 위치의 값을 이전 위치로 이동합니다.
            arr[j+1] = arr[j]
            j -= 1  # 이전 위치로 이동합니다.
        arr[j+1] = key  # key 값을 현재 위치에 저장합니다.
    return arr  # 정렬된 배열을 반환합니다.

'''
import numpy as np
import matplotlib.pyplot as plt

def draw(arr):
    plt.clf()   # 그래프 창을 초기화합니다.
    bars = plt.bar(range(len(arr)), arr)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, str(int(height)), ha='center', va='bottom')
    plt.draw()
    plt.pause(0.1)

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
            draw(arr)
        arr[j+1] = key
    draw(arr)
    return arr

# arr = [64, 25, 12, 22, 11]
arr = np.random.randint(1, 50, 20)
insertion_sort(arr)
plt.show()

'''