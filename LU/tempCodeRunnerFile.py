import numpy as np
from fractions import Fraction

def format_fraction_list(fraction_list):
    """리스트의 Fraction 객체를 a/b 형태의 문자열로 변환"""
    formatted_fractions = [format_fraction(frac) for frac in fraction_list]
    return ', '.join(formatted_fractions)

def format_fraction(fraction):
    """Fraction 객체를 a/b 형태의 문자열로 변환"""
    if fraction.denominator == 1:
        return str(fraction.numerator)
    else:
        return f"{fraction.numerator}/{fraction.denominator}"

class InvertibleMatrix:
    def __init__(self, size=3, value_range=(-10, 10)):
        """InvertibleMatrix 클래스 초기화"""
        self.size = size
        self.value_range = value_range
        self.matrix = self.generate_invertible_matrix()  # 가역 행렬 생성

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

class LUDecompositionWithEliminationMatrix:
    def __init__(self, matrix):
        """
        LU 분해를 소거 행렬을 사용하여 구현
        :param matrix: LU 분해할 행렬
        """
        self.matrix = np.array([[Fraction(num) for num in row] for row in matrix], dtype=object)  # 분수로 변환된 계수 행렬
        self.size = len(matrix)  # 행렬의 크기
        self.L = np.eye(self.size, dtype=object)  # 하삼각 행렬 (소거 행렬의 역행렬 곱으로 구성)
        self.U = self.matrix.copy()  # 상삼각 행렬 (가우스 소거로 계산)

        self.steps = []  # 각 단계의 소거 행렬 및 U 행렬 상태를 저장할 리스트

    def elimination_matrix(self, i, j, factor):
        """
        소거 행렬을 생성
        :param i: 소거 행렬에서 피벗 행의 인덱스
        :param j: 소거 행렬에서 제거할 행의 인덱스
        :param factor: 제거를 위한 스케일링 팩터
        :return: 소거 행렬
        """
        E = np.eye(self.size, dtype=object) #np.eye -> 단위행렬 생성, dtype=object: 업캐스팅 활용해서 fraction객체 넣을 수 있음.
        E[j, i] = -factor
        return E

    def inverse_elimination_matrix(self, i, j, factor):
        """
        소거 행렬의 역행렬을 생성
        :param i: 소거 행렬에서 피벗 행의 인덱스
        :param j: 소거 행렬에서 제거할 행의 인덱스
        :param factor: 제거를 위한 스케일링 팩터
        :return: 소거 행렬의 역행렬
        """
        E_inv = np.eye(self.size, dtype=object)
        E_inv[j, i] = factor
        return E_inv

    def decompose(self):
        """소거 행렬을 이용한 LU 분해 수행"""
        for i in range(self.size):
            for j in range(i + 1, self.size):
                factor = self.U[j, i] / self.U[i, i]
                E = self.elimination_matrix(i, j, factor)
                
                # U 행렬 갱신 (소거 행렬을 적용)
                self.U = E @ self.U
                # 소거 행렬의 역행렬을 이용해 L 행렬 갱신
                E_inv = self.inverse_elimination_matrix(i, j, factor)
                self.L = self.L @ E_inv
                
                # 각 단계에서 소거 행렬과 U, L 행렬 저장
                self.steps.append({
                    "elimination_matrix": E,
                    "E_inv": E_inv,
                    "updated_U": self.U.copy(),
                    "updated_L": self.L.copy()
                })

    def display_steps(self):
        """소거 행렬과 각 단계의 LU 분해 상태 출력"""
        for step_num, step in enumerate(self.steps, 1):
            print(f"\nStep {step_num}:")
            print("소거 행렬 E:")
            for row in step['elimination_matrix']:
                print(format_fraction_list(row))
            print("\n소거 행렬의 역행렬 E_inv:") # 소거행렬의 역행렬은 소거 행렬의 원래 연산을 되돌리는 행렬
            for row in step['E_inv']:
                print(format_fraction_list(row))
            print("\n업데이트된 U 행렬:")
            for row in step['updated_U']:
                print(format_fraction_list(row))
            print("\n업데이트된 L 행렬:")
            for row in step['updated_L']:
                print(format_fraction_list(row))

    def display_LU(self):
        """최종 L과 U 행렬을 출력"""
        print("\n최종 L 행렬:")
        for row in self.L:
            print(format_fraction_list(row))

        print("\n최종 U 행렬:")
        for row in self.U:
            print(format_fraction_list(row))

# 가역 행렬 생성
matrix_instance = InvertibleMatrix()
Matrix = matrix_instance.matrix
matrix_instance.display_matrix()

# LU 분해
lu_decomposition = LUDecompositionWithEliminationMatrix(Matrix)
lu_decomposition.decompose()
lu_decomposition.display_steps()
lu_decomposition.display_LU()
