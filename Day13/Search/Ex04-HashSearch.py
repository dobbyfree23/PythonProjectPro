'''


'''

class HashTable:
    def __init__(self, size):
        self.size = size
        self.hash_table = [[] for _ in range(self.size)]

    def hash_func(self, key):
        return key % self.size

    def insert(self, key, value):
        hash_value = self.hash_func(key)
        for i in range(len(self.hash_table[hash_value])):
            if self.hash_table[hash_value][i][0] == key:
                self.hash_table[hash_value][i][1] = value
                return
        self.hash_table[hash_value].append([key, value])

    def search(self, key):
        hash_value = self.hash_func(key)
        for i in range(len(self.hash_table[hash_value])):
            if self.hash_table[hash_value][i][0] == key:
                return self.hash_table[hash_value][i][1]
        return None

    def delete(self, key):
        hash_value = self.hash_func(key)
        for i in range(len(self.hash_table[hash_value])):
            if self.hash_table[hash_value][i][0] == key:
                del self.hash_table[hash_value][i]
                return True
        return False
    
    
# 실행코드

# 해시 테이블을 생성한다
hash_table = HashTable(10)

# 원소를 해시 테이블에 삽입한다
hash_table.insert(1, "apple")
hash_table.insert(2, "banana")
hash_table.insert(3, "cherry")

# 원소를 해시 테이블에서 검색한다
result1 = hash_table.search(1)
result2 = hash_table.search(2)
result3 = hash_table.search(3)
print(result1, result2, result3) # "apple" "banana" "cherry"