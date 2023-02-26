'''
힙정렬(Heap Sort)
    최대/최소 힙(heap) 자료구조를 이용하여 배열을 정렬하는 알고리즘으로,
    평균 시간 복잡도와 최악 시간 복잡도가 O(n log n)입니다.
'''

def heap_sort(arr):
    """
    힙 정렬 알고리즘을 사용하여 주어진 배열을 정렬합니다.

    Args:
    - arr: 정렬할 배열

    Returns:
    - 정렬된 배열
    """
    # 힙을 만들어줍니다.
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 정렬을 수행합니다.
    for i in range(n - 1, 0, -1):
        # 현재 루트 노드(최댓값)를 마지막 노드와 교환합니다.
        arr[0], arr[i] = arr[i], arr[0]

        # 최댓값을 제외한 나머지 노드들을 다시 힙으로 만들어줍니다.
        heapify(arr, i, 0)

    return arr


def heapify(arr, n, i):
    """
    주어진 배열에서 i번째 인덱스를 루트로 하는 서브트리를 힙으로 만듭니다.

    Args:
    - arr: 배열
    - n: 서브트리를 만들 범위(배열의 길이)
    - i: 루트 노드 인덱스

    Returns:
    - 없음
    """
    largest = i  # 가장 큰 원소의 인덱스
    left = 2 * i + 1  # 왼쪽 자식 노드 인덱스
    right = 2 * i + 2  # 오른쪽 자식 노드 인덱스

    # 왼쪽 자식 노드가 범위 내에 있고, 루트보다 크다면 largest를 왼쪽 자식 노드로 바꿉니다.
    if left < n and arr[largest] < arr[left]:
        largest = left

    # 오른쪽 자식 노드가 범위 내에 있고, largest보다 크다면 largest를 오른쪽 자식 노드로 바꿉니다.
    if right < n and arr[largest] < arr[right]:
        largest = right

    # largest가 i가 아니라면 루트 노드를 largest와 교환하고,
    # 교환한 노드를 루트로 하는 서브트리를 힙으로 만들어줍니다.
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# 실행코드
arr = [4, 10, 3, 5, 1]
sorted_arr = heap_sort(arr)
print(sorted_arr)
