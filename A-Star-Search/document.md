## 作业1

<table>
<thead>
  <tr>
    <th>迷宫</th>
    <th>算法</th>
    <th>用时/s</th>
    <th>展开节点数</th>
    <th>路径代价</th>
    <th>分数</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">tinyMaze</td>
    <td>dfs</td>
    <td>0.0</td>
    <td>15</td>
    <td>8</td>
    <td>502.0</td>
  </tr>
  <tr>
    <td>bfs</td>
    <td>0.0</td>
    <td>16</td>
    <td>8</td>
    <td>502.0</td>
  </tr>
  <tr>
    <td>A*</td>
    <td>0.0</td>
    <td>14</td>
    <td>8</td>
    <td>502.0</td>
  </tr>
  <tr>
    <td rowspan="3">smallMaze</td>
    <td>dfs</td>
    <td>0.0</td>
    <td>93</td>
    <td>37</td>
    <td>473.0</td>
  </tr>
  <tr>
    <td>bfs</td>
    <td>0.0</td>
    <td>94</td>
    <td>19</td>
    <td>491.0</td>
  </tr>
  <tr>
    <td>A*</td>
    <td>0.0</td>
    <td>53</td>
    <td>19</td>
    <td>491.0</td>
  </tr>
  <tr>
    <td rowspan="3">mediumMaze</td>
    <td>dfs</td>
    <td>0.0</td>
    <td>269</td>
    <td>246</td>
    <td>264.0</td>
  </tr>
  <tr>
    <td>bfs</td>
    <td>0.0</td>
    <td>275</td>
    <td>68</td>
    <td>442.0</td>
  </tr>
  <tr>
    <td>A*</td>
    <td>0.0</td>
    <td>224</td>
    <td>68</td>
    <td>442.0</td>
  </tr>
  <tr>
    <td rowspan="3">bigMaze</td>
    <td>dfs</td>
    <td>0.0</td>
    <td>466</td>
    <td>210</td>
    <td>300.0</td>
  </tr>
  <tr>
    <td>bfs</td>
    <td>0.0</td>
    <td>620</td>
    <td>210</td>
    <td>300.0</td>
  </tr>
  <tr>
    <td>A*</td>
    <td>0.0</td>
    <td>549</td>
    <td>210</td>
    <td>300.0</td>
  </tr>
</tbody>
</table>
## 作业2

假设当前位置为 $P_0$，当前食物位置的集合为 $F$，$D(A,B)$ 是从 $A$ 点到 $B$ 点的最短路径的长度，则启发函数定义为：
$$
h(P_0) = \max_{F_i\in F}D(P_0,F_i)
$$
显然 $h(P_0)\geq 0$。设 $F_m=\mathrm{argmax}\ h(P_0)$，即离吃豆人当前位置最远的食物位置。首先，吃豆人必须至少到达 $F_m$ 处才有可能吃完所有豆子（结束），由于中间还要吃其他豆子，故显然 $h(P_0)\leq h^*(P_0)$，即 $h(P_0)$ 是可采纳的。

设 $P_0'$ 是除 $P_0$ 和 $F_m$ 之外的另外一点，$F_m=\mathrm{argmax}\ h(P_0')$。

由于 $D(P_0,F_m)$ 是最短路径，故 $D(P_0,F_m) \leq D(P_0,P_0') + D(P_0',F_m)$。而 $F_m'$ 是距离 $P_0'$ 最远的食物，故 $D(P_0',F_m)\leq D(P_0',F_m')$。

故 $D(P_0,F_m)\leq D(P_0,P_0')+D(P_0',F_m')$，即 $h(P_0)\leq D(P_0,P_0')+h(P_0')$ ，即 $h(P_0)$ 具有一致性。

## 作业3

迷宫见 `myMaze.lay` 文件，录屏见 `myMaze.avi` 文件。

P.S. 算法加入了一个 trick：当前所有步骤在 depth 范围内效果相同时，由于算法的模式是固定的，pacman 可能会呆在原地或反复徘徊，于是改为让 pacman 随机选择一个方向。