'''
선형 검색(Linear Search)
        간단한 검색 알고리즘으로, 데이터를 처음부터 끝까지
        하나씩 차례대로 비교하여 원하는 값을 찾는다.
'''

def linear_search(arr, target):
    for i in range(len(arr)):
        # 현재 위치의 원소가 탐색 대상인 경우 인덱스를 반환한다
        if arr[i] == target:
            return i
    # 모든 원소를 탐색한 경우 탐색 대상을 찾지 못한 것으로 간주하고 -1을 반환한다
    return -1


arr = [1, 3, 5, 7, 9]
target = 5
result = linear_search(arr, target)
print(result) # 2