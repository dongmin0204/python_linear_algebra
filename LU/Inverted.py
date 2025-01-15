import numpy as np
from fractions import Fraction

class InvertibleMatrix:
    def __init__(self, size=3, value_range=(-10, 10)):
        """InvertibleMatrix 클래스 초기화"""
        self.size = size
        self.value_range = value_range
        self.matrix = self.generate_invertible_matrix()  # 가역 행렬 생성
        self.fixed_matrix = [[-1,0,-1],[-1, 0, 0],[2, 1, 2]]
    def generate_invertible_matrix(self):
        """가역 행렬을 무작위로 생성. 행렬식이 0이 아닐 때까지 반복하여 생성"""
        while True:
            matrix = np.random.randint(self.value_range[0], self.value_range[1] + 1, size=(self.size, self.size))
            if np.linalg.det(matrix) != 0:
                return matrix

    def display_matrix(self):
        """생성된 행렬을 출력"""
        print("원본 행렬 A:")
        for row in self.matrix:
            print(row)

def gauss_jordan_inverse(matrix):
    """가우스-조던 소거법을 사용하여 행렬의 역행렬을 계산"""
    n = len(matrix)
    # 입력 행렬을 Fraction 객체로 변환
    A = np.array([[Fraction(num) for num in row] for row in matrix], dtype=object)

    # 단위 행렬을 추가하여 확장 행렬 [A | I] 생성
    augmented_matrix = np.hstack([A, np.eye(n, dtype=object)])
    
    # 확장 행렬 초기 상태 출력
    print("\n확장 행렬 초기 상태 [A | I]:")
    display_matrix(augmented_matrix)

    # 가우스-조던 소거법 적용
    for i in range(n):
        # 피벗 요소가 0이면 행 교환
        if augmented_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    print(f"\n행 {i}과 행 {j}을 교환했습니다.")
                    display_matrix(augmented_matrix)
                    break
            else:
                raise ValueError("역행렬을 구할 수 없습니다. 행렬이 가역이 아닙니다.")

        # 피벗 행을 1로 만듦
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot
        print(f"\n행 {i}을(를) 피벗 {pivot}으로 나눴습니다:")
        display_matrix(augmented_matrix)

        # 다른 행에서 피벗 열을 0으로 만듦
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
                print(f"\n행 {j}에서 {factor} * 행 {i}을 뺐습니다:")
                display_matrix(augmented_matrix)

    # 오른쪽 부분은 역행렬
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix

def display_matrix(matrix, title="Matrix"):
    """행렬 출력 함수"""
    print(f"\n{title}:")
    for row in matrix:
        print([str(x) for x in row])

# 테스트용 코드
def perform_gauss_jordan_inverse(matrix_size=3, value_range=(-5, 5)):
    # 가역 행렬 생성
    matrix = InvertibleMatrix(matrix_size, value_range).fixed_matrix

    print("\n원본 행렬 A:")
    display_matrix(matrix)

    # 가우스-조던 소거법을 사용하여 역행렬 계산
    try:
        A_inv = gauss_jordan_inverse(matrix)
        print("\nA의 역행렬:")
        display_matrix(A_inv, "Inverse Matrix")
    except ValueError as e:
        print(f"오류 발생: {e}")

# 실행
perform_gauss_jordan_inverse()
