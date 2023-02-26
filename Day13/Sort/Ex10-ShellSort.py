'''
셸 정렬 (Shell sort)

'''


def shell_sort(arr):
    # 셸 정렬에서 사용할 간격 값을 계산한다
    gap = len(arr) // 2
    # 간격을 점점 줄여가며 삽입 정렬을 적용한다
    while gap > 0:
        for i in range(gap, len(arr)):
            temp = arr[i]
            j = i
            while j >= gap and arr[j-gap] > temp:
                arr[j] = arr[j-gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    # 정렬된 결과를 반환한다
    return arr