'''
보간 검색(Interpolation Search)
    이진 검색을 보완한 검색 알고리즘

    값의 분포에 따라 중앙값의 위치를 예측하며,
    예측한 위치에 있는 값이 탐색 대상보다
    큰지 작은지에 따라 탐색 범위를 좁혀가며 탐색합니다.

    이 알고리즘은 값의 분포가 고르지 않은 경우에는
    이진 검색보다 성능이 떨어질 수 있습니다.

이진 탐색
pos = (low + high)/2

보간 탐색

배열의 중간값(mid)과 찾으려는 값(value)의 상대적인 위치 비율을 계산합니다.
position = (value - min_value) / (max_value - min_value)
배열의 중간값(mid)을 다음과 같은 공식으로 계산합니다:
mid = left + int((right - left) * position)
'''
def interpolation_search(arr, target):
    # 탐색 범위의 시작점과 끝점을 지정한다
    left, right = 0, len(arr) - 1
    # 탐색 범위가 존재하는 동안 반복한다
    while left <= right:
        # 값의 분포에 따라 중앙값의 위치를 예측한다
        pos = left + int((float(right - left) / (arr[right] - arr[left])) * (target - arr[left]))

        print("pos : ",pos)
        print("target : ",target)

        # 예측한 위치가 배열 범위를 벗어나는 경우, 탐색 대상이 없다고 판단한다
        if pos < 0 or pos >= len(arr):
            return -1
        # 예측한 위치에 탐색 대상이 있는 경우, 인덱스를 반환한다
        if arr[pos] == target:
            return pos
        # 예측한 위치에 탐색 대상보다 큰 값이 있는 경우, 오른쪽 부분 배열을 탐색한다
        elif arr[pos] < target:
            left = pos + 1
        # 예측한 위치에 탐색 대상보다 작은 값이 있는 경우, 왼쪽 부분 배열을 탐색한다
        else:
            right = pos - 1
    # 탐색 대상을 찾지 못한 경우 -1을 반환한다
    return -1


arr = [1, 3, 5, 7, 9,12,15,17,22,28,99,101,106]
target =22
result = interpolation_search(arr, target)
print(result) # 2