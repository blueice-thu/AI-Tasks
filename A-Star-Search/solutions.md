# hw1

姓名：沈澎博    班级：软件82    学号：2017011672

## 第一题

(1) 正确。当搜索树各个相邻节点间的距离均相等时，UCS 退化为 BFS。如下图图 1 所示，用 UCS 算法搜索从 A 到 H 的最短路径，依次搜索的顺序是：A, B, C, D, E, F, G, H，即为 BFS 的搜索顺序。

<img src=".\imgs\1.png" alt="1" style="zoom: 67%;" />

(2) 正确。对于连通的无向有限图 $G(V,E)$ 使用 BFS 算法，设 $V_1\subseteq V$ 为已经搜索过的点的集合，$V_2\subseteq V$ 为未搜索过的点的集合，则 $V=V_1+V_2$。在 BFS 的迭代中，会不断从 $V_2$ 中选择 $v_i\in V_1$ 的相邻结点，并转移到 $V_1$ 中。由于 $G$ 是连通的，故 $V_1$ 中元素最终将全部转移到 $V_2$ 中，当 $V_2$ 为空时结束，即所有点都将被遍历一次。因此 BFS 算法是完备的。

(3) 正确。DFS 可能会陷入无限循环。例如，在一个有环的无向有限图中，DFS 可能会不停在一个环路中搜索。如图 2 所示，DFS 可能在 A - B - C - D - A 中循环，因此找不到从 A 到 E 的路径。

(4) 正确。$A^*$ 搜索相对 UCS 引入了启发式函数 $h(n)$，如果设定 $h(n)=0$，则 $f(n)=g(n)$，$A^*$ 搜索退化为 UCS。

## 第二题

(1) 设该 CSP 问题为 $P(X,D,C)$

变量：$X=\{(X_1,Y_1),(X_2,Y_2),...,(X_n,Y_n)\}$，其中 $(X_i,Y_i)$ 代表第 $i$ 个马位于第 $X_i$ 行第 $Y_i$ 列。

值域：$D=\{(1,1),(1,2),...,(1,n),(2,1),...,(n,n)\}$，即变量可以取棋盘上 $n^2$ 个位置中的一个。

$C$ 为约束。

(2) 受到的约束有：

- 任意两个马不能放在同一位置，即：$\forall i\ne j, |X_i- X_j|+ | Y_i- Y_j|\ne 0$；
- 任意两个马不能位于可以互相攻击的位置，即：$\forall i\ne j$，若 $|X_i-X_j|\times |Y_i-Y_j|\ne 2$。

(3) 

```
FUNCTION Place-Horse(problem) returns a state that is local maximum
	assign a different random value for every horse
	current <- initial status
	calculate confict number of every position
    confict <- sum of all confict horses
	while true do
		moved_horse <- the horse has maximum confict number
		(newX, newY) <- the position has minimum confict number
		neighbor <- set position of moved_horse be (newX, newY)
		calculate confict number of every position
		new_confict <- sum of all confict horses
		if new_confict > conflict
			return current
		current <- neighbor
		confict <- neighbor
```

## 第三题

为每个结点的每个元素记录一个是否可用的布尔状态，初始时均为可用。用 $Q$ 表示二元组（节点，元素）的队列，在一次循环中，从 $Q$ 中选择并移除一个二元组 $(v_i,a)$。对于每个通过可用状态的元素 $d$ 与 $(v_i,a)$ 相容的邻居 $v_j$，减少 $(v_j,d,v_i)$ 的计数值 1 。如果该计数值变为 0，则将 $(v_j,d)$ 加入到 $Q$ 中，并将 $v_j$ 结点的 元素 $d$ 标记为不可用，进入下一次循环。
