'''
기수 정렬 (Radix sort)

'''

def radix_sort(arr):
    # 입력된 배열의 최대값을 찾는다
    max_value = max(arr)
    # 최대값의 자리수를 구한다
    digit = len(str(max_value))
    # 각 자리수에 대해 counting sort를 적용한다
    for i in range(digit):
        # 입력된 배열의 각 원소를 i번째 자리수를 기준으로 counting sort를 적용한다
        arr = counting_sort_by_digit(arr, i)
    # 정렬된 결과를 반환한다
    return arr

def counting_sort_by_digit(arr, digit):
    # 0부터 9까지의 counting 배열을 생성한다
    count = [0] * 10
    # 입력된 배열의 각 원소의 자리수를 기준으로 counting 배열에 저장한다
    for num in arr:
        count[get_digit(num, digit)] += 1
    # count 배열을 순회하며 누적합을 구한다
    for i in range(1, len(count)):
        count[i] += count[i-1]
    # 출력할 배열을 생성하고, count 배열을 역순으로 순회하며 원소를 채운다
    result = [0] * len(arr)
    for num in reversed(arr):
        index = get_digit(num, digit)
        result[count[index]-1] = num
        count[index] -= 1
    # 정렬된 결과를 반환한다
    return result

def get_digit(num, digit):
    return (num // 10**digit) % 10