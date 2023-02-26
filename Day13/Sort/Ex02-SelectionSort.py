'''
파일명 : Ex02-SelectionSort.py
    주어진 리스트에서 최소값을 찾아 맨 앞에 있는 값과 교체하는 알고리즘
    O(n^2)의 시간 복잡도
'''
def selection_sort(arr):
    # 배열을 정렬하는 함수를 구현한다
    for i in range(len(arr)):
        # i번째 인덱스를 최소값으로 가정한다
        min_idx = i
        # i+1부터 시작하여 arr의 길이까지 반복하며 최소값의 인덱스를 찾는다
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                # 최소값보다 더 작은 값을 찾으면, 그 값을 최소값으로 설정한다
                min_idx = j
        # 최소값의 인덱스와 i번째 인덱스의 값을 교환한다
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    # 정렬된 배열을 반환한다
    return arr