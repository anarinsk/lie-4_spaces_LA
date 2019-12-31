**Four Fundamental Spaces of Linear Algebra**

2019-12-31
Jun Sok Huhh | :house:[lostineconomics.com](http://lostineconomics.com)


# Tales of Two Lines 

행렬을 행 공간으로 이해하는 것과 열 공간으로 이해하는 것은 같은 해를 구하는 문제에서도 전혀 다른 함의를 지닌다.  예를 들어보자. 

아래의 연립 방정식을 풀고 싶다고 하자. 

$$
\begin{aligned}
2 x + y & = 3\\
x - 2y & = -1
\end{aligned}
$$

행렬로 나타내면 다음과 같다. 

$$
\begin{bmatrix}
2 & 1 \\
1 & -2
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} = 
\begin{bmatrix}
3 \\
-1
\end{bmatrix} 
$$

## Row picture 

행으로 이해해보자. 이게 사실 우리에게 익숙한 방식이다. 이 차원 직선 두 개를 그리고 그 교점을 찾으면 되다. 이게 위 문제를 행으로 보는 관점이다. 아래 그림을 참고하자.[^1] (조교 분 모습을 잘라내지 못한 점 양해 부탁) 

[^1]: 그림의 출처는 [여기](https://www.youtube.com/watch?v=My5w4MXWBew)

<p align="center"><kbd>
  <img src="https://github.com/anarinsk/lie-4_spaces_LA/blob/master/assets/imgs/row-picture.png?raw=true" width="600">
</kbd></p>


## Column picture 

이제 이 문제를 컬럼으로 보자. 행렬을 열로 보면, $(2 \times 1)$ 벡터다. 이 벡터를 좌표로 나타나면 이제 $x$, $y$는 식의 방정식 우변의 벡터를 얻는 데 필요한 두 행 벡터에 가중치를 부여하는 역할을 한다. 아래 그림을 보자. 

<p align="center"><kbd>
  <img src="https://github.com/anarinsk/lie-4_spaces_LA/blob/master/assets/imgs/column-picture.png?raw=true" width="600">
</kbd></p>

## Which of two? 

둘 다 쓸모가 있는 관점이지만 열 공간으로 보는 관점이 몇 가지 점에서 수학적으로 좋다. 우선, 열 공간으로 보게 되면 계산에 동원되는 모든 대상들이 벡터 공간에 위치하게 된다. 벡터 공간은 반드시 $\boldsymbol{0}$를 포함하게 된다. 행 공간에서는 이게 가능하다. 투입과 산출이 모두 벡터로 표현되고 투입의 선형 결합을 통해 산출을 표현하게 되는 구조다. 그런데, 열 공간의 관점에서는 이런 벡터 공간의 수학적인 결과를 활용하기 힘들다. 

# Big Picture of Linear Algebra 

기본적으로 행렬은 함수다. 즉, $(m \times n)$ 행렬이 있다면, 이는 $(n \times 1)$의 투입 벡터를 $(m \times 1)$의 산출 벡터로 바꾸는 역할을 한다. $A x = b$는 $f(x) =b$로 이해하면 좋다. 다만 차원이 임의의 자연수일 뿐이다. 

마찬가지로 $A^T$는 $(m \times 1)$ 투입 벡터를 $(n \times 1)$ 산출 벡터로 바꾸는 역할을 한다. 이들 사이에 어떤 관계는 어떨까? 이를 나타내는 것이 길버트 스트랭(Gibert Strang) 선생이 말한 선형대수의 큰 그림이다. 아래 그림을 보자. 

<p align="center"><kbd>
  <img src="https://i.stack.imgur.com/dfZND.png" width="600">
</kbd></p>

그림 자체로 그냥 이해가 간다. 열 공간으로 이해하는 습관이 들었다면, 그림이 뒤집혀야 하지 않나, 싶겠지만 $A x = b$의 형태로 먼저 이해하면 좋다.  

## Row space 

- $A$의 행 공간은 $\mathbb{R}^n$에 속한다. 

$$
A = 
\begin{bmatrix}
r_1^T \\
\vdots\\
r_m^T
\end{bmatrix} 
$$

- $r_i$는 $A$의 $i$ 번째 행을 원소로 지닌 열 벡터이고 $(n \times 1)$이다.  
- 행 공간의 영 공간(null space) 역시 $\mathbb{R}^n$에 속한다. 영 공간이란 $A x = \boldsymbol{0}$을 만족하는 $x \neq \boldsymbol{0}$의 벡터이므로 이 역시 $(n \times 1)$ 벡터다. 

### Orthogonality of row space and null space 

두 벡터는 직교할까? 행 공간 $\mathcal{R}$의 정의는 다음과 같다. 

$$
\mathcal{R}(A) = \{  x_r \in \mathbb{R}^n \vert x_r = \sum_{i=1}^{m} r_i \alpha_i ,~\text{where}~ \alpha_i \in \mathbb{R}, ~r_i \in \mathbb{R}^n \}
$$

영공간(nullspace)에 속하는 벡터를 $x_n$라고 할 때(notation에 약간의 교란이 발생하지만 그림과의 일치를 위해 일단 이렇게 표기하도록 하자), 영공간의 정의에 따라서 $r_i^T x_n = 0$. 

$$
{x_r^T} x_n =  \sum_{i=1}^{m} \alpha_i (r_i^T x_n)  = 0
$$

그리고 그림에서 보듯이 다음과 같은 관계가 성립한다. 

- $A x_r = \underset{(m \times 1)}{b_c}$
- $A x_n = \boldsymbol{0}$
- $A (x_r + x_n) = b_c$
- $x_r^T x_n = \boldsymbol{0}$

위 관계에서 $b_c$, $\boldsymbol{0}$는 모두 열 공간에 존재하는 벡터들이므로 $(m \times 1)$의 크기를 지닌다는 점에 유의하자. 

## Column space 

- $A$의 열 공간은 $\mathbb{R}^m$에 속한다.  

$$
\begin{bmatrix}
c_1, ~\cdots~, c_n
\end{bmatrix}
$$

- 열 공간의 영 공간, 좌 영공간(left nullspace) 역시 $\mathbb{R}^m$에 속한다. 이는 $A^T x = 0$에 의해 정의된다. 
- 나머지 과정은 비슷하게 전개할 수 있다. 열 공간 $\mathcal{C}$의 정의는 다음과 같다. 

$$
\mathcal{C}(A) = \{  x_c \in \mathbb{R}^m \vert x_c = \sum_{i=1}^{n} c_i \alpha_i ,~\text{where}~ \alpha_i \in \mathbb{R}, ~c_i \in \mathbb{R}^m \}
$$

좌 영공간의 정의에 따르면, $c_i^T x_n = 0$가 성립한다. 따라서, 

$$
{x_c^T} x_n =  \sum_{i=1}^{n} \alpha_i (r_i^T x_n)  = 0
$$

- $A^T x_c = \underset{(n \times 1)}{b_r}$
- $A^c x_n = \boldsymbol{0}$
- $A (x_c + x_c) = b_r$
- $x_r^T x_n = \boldsymbol{0}$

## Exchange of row and column 

$A^T$의 열 공간이 곧 $A$의 행 공간이 된다. 따라서 $\mathcal{R}(A) = \mathcal{C}(A^T)$가 된다.
 
$$
A^T =  
\begin{bmatrix}
r_1 ~,\dotsc, ~ r_m
\end{bmatrix} 
$$

위의 그림을 컬럼 스페이스로만 다시 표현하면 다음과 같다. 

<p align="center"><kbd>
  <img src="https://github.com/anarinsk/lie-4_spaces_LA/blob/master/assets/imgs/fundamental.png?raw=true" width="600">
</kbd></p>

# Why? 

이 네 개의 스페이스가 맺고 있는 관련성은 그 자체만으로도 중요하고 흥미로운 것이지만, 이를 통해 이른바 SVD(Singluar Value Decomposition)을 달성할 수 있다. 

<p align="center"><kbd>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Singular_value_decomposition_visualisation.svg/1920px-Singular_value_decomposition_visualisation.svg.png" width="400">
</kbd></p>

먼저 매트릭스 $A$의 열 공간에 속하는 원소 중에서 $r$ 개만 서로 독립이라고 하자. 이렇다면 이 성분으로만 구성된 매트릭스 $U$를 만들 수 있다. 이때 매트릭스 $U$의 켤레 전치행렬은 $U^*$라고 하면, $U U^* = I_m$이 성립한다. 그리고, 행 공간에 속하는 원소 역시 $r$ 개만 독립이고, 이를 기반으로 $V$를 만들 수 있다. 그리고 이 사이에 특성값(singular value)을 대각행렬로 지니는 $\Sigma$를 넣으면 $A$는 다음과 같이 세 가지로 분해된다. 

<p align="center"><kbd>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Singular-Value-Decomposition.svg/1920px-Singular-Value-Decomposition.svg.png" width="400">
</kbd></p>

기본적으로 행렬은 함수다. 즉 어떤 벡터의 변형이다. $M$에 투입되는 $(n \times 1)$의 벡터 $x$가 있다고 하자.  

1. 벡터의 방향을 돌린다 ($V^*$)  
2. 특성값 행렬($\Sigma$)로 차원을 바꾸면서 각 축의 방향을 조절한다.  
3. 마지막으로 $U$를 통해서 차원을 바꿔준다. 
<br>
<br>
<br>
 :house:[lostineconomics.com](http://lostineconomics.com) | Jun Sok Huhh
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMjIyNzkwNTgsLTgwMDQxMjY2OV19
-->