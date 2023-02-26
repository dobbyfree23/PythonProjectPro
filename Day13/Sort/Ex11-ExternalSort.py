# 예시로 사용된 정렬 알고리즘의 내용은 생략합니다.
# 이 코드는 대용량의 데이터를 처리하기 위한 외부 정렬 알고리즘의 전체적인 구조를 보여줍니다.
def external_sort(file_path, buffer_size):
    # 대용량의 데이터를 여러 개의 작은 블록으로 나누어 정렬한다
    block_size = buffer_size // 2
    sorted_blocks = []
    with open(file_path, 'r') as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            block = list(map(int, block.split()))
            block = merge_sort(block)
            sorted_blocks.append(block)
    # 작은 블록들을 조합하여 최종 결과를 생성한다
    result = []
    heap = [(block[0], i) for i, block in enumerate(sorted_blocks) if block]
    heapq.heapify(heap)
    while heap:
        value, index = heapq.heappop(heap)
        result.append(value)
        block = sorted_blocks[index]
        block.pop(0)
        if block:
            heapq.heappush(heap, (block[0], index))
    # 정렬된 결과를 반환한다
    return result