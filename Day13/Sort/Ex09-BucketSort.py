'''
버킷정렬(BucketSort)
'''

def bucket_sort(arr, bucket_size=5):
    # 입력된 배열의 최소값과 최대값을 찾는다
    min_value, max_value = min(arr), max(arr)
    # 버킷의 개수를 계산한다
    bucket_count = (max_value - min_value) // bucket_size + 1
    # 빈 버킷을 생성한다
    buckets = [[] for _ in range(bucket_count)]
    # 입력된 배열의 각 원소를 버킷에 분배한다
    for num in arr:
        index = (num - min_value) // bucket_size
        buckets[index].append(num)
    # 각 버킷을 정렬하고, 결과 배열에 추가한다
    result = []
    for bucket in buckets:
        result += sorted(bucket)
    # 정렬된 결과를 반환한다
    return result
