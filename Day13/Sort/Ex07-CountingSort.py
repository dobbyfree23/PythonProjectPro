'''
계수 정렬(Counting Sort)은
    정수나 정수로 표현할 수 있는 자료에 대해서만 적용할 수 있는 비교 기반 정렬 알고리즘입니다.
    알고리즘은 입력 배열을 한 번 순회하며 각 숫자의 빈도수를 구합니다.
    이후, 누적합을 구하고, 뒤에서부터 순회하며 원래 배열의 순서를 유지하면서 정렬합니다.
    계수 정렬은 상대적으로 빠른 수행 속도를 가지고 있으며, 입력값의 크기에 비례하는 메모리 공간이 필요합니다.

'''

def counting_sort(arr):
    # 입력된 배열의 최대값을 찾는다
    max_value = max(arr)
    # 입력된 배열의 최대값+1 길이의 0으로 채워진 배열을 생성한다
    count = [0] * (max_value+1)
    # 입력된 배열의 각 원소의 개수를 count 배열에 저장한다
    for num in arr:
        count[num] += 1
    # count 배열을 순회하며 누적합을 구한다
    for i in range(1, len(count)):
        count[i] += count[i-1]
    # 출력할 배열을 생성하고, count 배열을 역순으로 순회하며 원소를 채운다
    result = [0] * len(arr)
    for num in reversed(arr):
        result[count[num]-1] = num
        count[num] -= 1
    # 정렬된 결과를 반환한다
    return result