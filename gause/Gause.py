import numpy as np
import random
from fractions import Fraction
def format_fraction_list(fraction_list):
    """리스트의 Fraction 객체를 a/b 형태의 문자열로 변환

    Args:
        fraction_list: Fraction 객체로 이루어진 리스트

    Returns:
        str: 변환된 문자열 (각 요소는 쉼표로 구분)
    """

    formatted_fractions = [format_fraction(frac) for frac in fraction_list]
    return ', '.join(formatted_fractions)

# 기존 format_fraction 함수는 내부에서 사용
def format_fraction(fraction):
    """Fraction 객체를 a/b 형태의 문자열로 변환

    Args:
        fraction: Fraction 객체

    Returns:
        str: 변환된 문자열
    """

    if fraction.denominator == 1:
        return str(fraction.numerator)
    else:
        return f"{fraction.numerator}/{fraction.denominator}"
# 가역(역행렬이 존재하는) 행렬을 생성하는 클래스
class InvertibleMatrix:
    def __init__(self, size=3, value_range=(-10, 10)):
        """
        InvertibleMatrix 클래스 초기화

        :param size: 행렬의 크기 (기본값은 3x3)
        :param value_range: 행렬 원소의 값 범위 (기본값은 -10에서 10 사이)
        """
        self.size = size
        self.value_range = value_range
        self.matrix = self.generate_invertible_matrix()  # 가역 행렬 생성

    def generate_invertible_matrix(self):
        """
        가역 행렬을 무작위로 생성. 행렬식이 0이 아닐 때까지 반복하여 생성.

        :return: 가역 행렬
        """
        while True:
            # 지정된 범위 내에서 무작위로 행렬 생성
            matrix = np.random.randint(self.value_range[0], self.value_range[1] + 1, size=(self.size, self.size))
            if np.linalg.det(matrix) != 0:  # 행렬식이 0이 아닌 경우(가역인 경우) 반환
                return matrix

    def display_matrix(self):
        """생성된 행렬을 출력"""
        print(self.matrix)


# 가우스 소거법을 사용하여 연립 방정식을 푸는 클래스
class GaussianElimination:
    def __init__(self, matrix, constants):
        """
        GaussianElimination 클래스 초기화
        
        :param matrix: 계수 행렬
        :param constants: 상수 벡터
        """
        self.matrix = [[Fraction(num) for num in row] for row in matrix]  # 계수 행렬을 분수 배열로 변환
        self.constants = [Fraction(num) for num in constants] # 상수 벡터를 분수 변환
        # 계수 행렬과 상수 벡터를 합쳐 확장 행렬 생성
        self.augmented_matrix = []
        for row, constant in zip(self.matrix, self.constants):
            self.augmented_matrix.append(row + [constant])
            
        self.steps = []  # 각 단계를 저장할 리스트
  


    def forward_elimination(self):
        """전진 소거 과정을 수행하여 상삼각 행렬로 변환"""
        n = len(self.matrix)
        cnt_1 = len(self.steps)
        for i in range(n):
            # 피봇팅: 가장 큰 값을 가진 행을 선택하여 현재 행과 교환하여 0으로 나누는 것을 방지
            max_row = max(range(i, n), key=lambda r: abs(self.augmented_matrix[r][i]))
            if i != max_row:
                # 행 교환
                self.augmented_matrix[i], self.augmented_matrix[max_row] = self.augmented_matrix[max_row], self.augmented_matrix[i]
                self.steps.append(f"{i+1} 행과 {max_row + 1}행을 교환:\n{self.augmented_matrix}")

            # 대각 원소를 1로 만들고 아래쪽 원소를 제거
            for j in range(i + 1, n):
                # 제거할 행의 비율 계산
                factor = Fraction(self.augmented_matrix[j][i], self.augmented_matrix[i][i])
                # 비율을 이용해 현재 행을 제거할 행에 빼기 연산 수행
                self.augmented_matrix[j] = [
                    element - factor * self.augmented_matrix[i][index]
                        for index, element in enumerate(self.augmented_matrix[j])
                    ]   
                self.steps.append(f"{i+1} 행과 {j+ 1}행을 제거:\n{self.augmented_matrix}")
        

    def back_substitution(self):
        """후진 대입을 통해 해를 구하는 과정"""
        n = len(self.matrix)
        x = np.zeros(n, dtype=int)  # 해를 저장할 배열
        for i in range(n - 1, -1, -1):
            # 현재 행의 상수에서 이미 구한 해를 빼고 나머지를 현재 변수의 계수로 나눔
            x[i] = (self.augmented_matrix[i][-1] - sum(Fraction(a) * b for a, b in zip(self.augmented_matrix[i][i + 1:n], x[i + 1:n]))) / Fraction(self.augmented_matrix[i][i])
            self.steps.append(f"후진 대입 단계 {n - i}:\n현재 해: [x,y,z] = {x}")
        return x

    def solve(self):
        """전체 과정을 통해 연립 방정식의 해를 구함"""
        self.forward_elimination()  # 전진 소거 과정
        solution = self.back_substitution()  # 후진 대입 과정
        return solution
    
    def display_steps(self):
        """가우스 소거법의 각 단계를 출력"""
        for step in self.steps:
            # 각 행을 분수 형태로 변환
            formatted_rows = [format_fraction_list(row) for row in self.augmented_matrix]
            print(step.replace(str(self.augmented_matrix), '\n'.join(formatted_rows)))
        solution = self.solve()
        print(f"최종 해: [{', '.join(format_fraction(x) for x in solution)}]")

    def display_step(self,step):
        """특정 단계를 출력"""
        formatted_rows = [format_fraction_list(row) for row in self.augmented_matrix]
        print(f"\n{self.steps[step].replace(str(self.augmented_matrix), '\n'.join(formatted_rows))}")


# 가역 행렬 생성
matrix_instance = InvertibleMatrix()
Matrix = matrix_instance.matrix
matrix_instance.display_matrix()

# 행렬과 정수 해를 갖도록 상수 벡터 생성
rand_const = [random.randint(-3, 5) for _ in range(3)]
constants = Matrix @ np.array(rand_const)  # 예를 들어 [1, 2, 3]를 곱하여 정수 해 보장


# 가우스 소거법을 통해 문제 해결
gaussian_solver = GaussianElimination(Matrix, constants)
print(constants) #상수 벡터
gaussian_solver.solve()  # 방정식 풀기
gaussian_solver.display_steps()