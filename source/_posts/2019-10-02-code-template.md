---
title: 机试模板整理
top: false
cover: false
toc: true
date: 2019-10-02 15:57:40
password:
summary:
tags:
- 保研
- 机试
- 编程
- 算法
categories: 
- 程序设计
---

> 这是当时我去参加清华计算机系夏令营时为机考准备的模板，部分参考链接在最后列出, 我整理了一些我认为常见的一些算法，顺便改了改部分代码风格（其实主要是加空格hhh）。最后机试的时候实际只用到了关于强连通分量缩点的那部分代码，说起来那部分代码还是机试前一天晚上在宾馆手抄的（果然我总能考前精准押题233333）。
>
> 今年清华机试一反常态难度大大降低，满分的似乎都有25%，我当时在测试数据上也是取得了满分（之后有空把原题也整理一下放上来）。不过最后因为其他一些原因，也没有去报名清华九月的推免，也就没能去成清华。不过这些算法模板作为学习材料还是很不错的（说不定以后什么时候又用上了）。

## 0. 头文件

```cpp
#define _CRT_SBCURE_NO_DEPRECATE
#include <set>
#include <cmath>
#include <queue>
#include <stack>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <functional>

using namespace std;

const int INF = 0x3f3f3f3f;
```



## 1. 经典算法

### 1.1. 埃拉托斯特尼筛法

```cpp
/*
    |埃式筛法|
    |快速筛选素数|
*/
int prime[maxn];  
bool is_prime[maxn];

int sieve(int n){
    int p = 0;
    for(int i = 0; i <= n; ++i)
        is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= n; ++i){   // 注意数组大小是n
        if(is_prime[i]){
            prime[p++] = i;
            for(int j = i + i; j <= n; j += i)  // 轻剪枝，j必定是i的倍数
                is_prime[j] = false;
        }
    }
    return p;   // 返回素数个数
}
```

### 1.2. 快速幂

```cpp
typedef long long LL;   // 视数据大小的情况而定

LL powerMod(LL x, LL n, LL m)
{
    LL res = 1;
    while (n > 0){
        if  (n & 1) // 判断是否为奇数，若是则true
            res = (res * x) % m;
        x = (x * x) % m;
        n  >>= 1;    // 相当于n /= 2;
    }
    return res;
}
```

### 1.3. 大数模拟

#### 1.3.1. 大数加法

```cpp
/*
    |大数模拟加法|
    |用string模拟|
*/
string add1(string s1, string s2)
{
    if (s1 == "" && s2 == "")   return "0";
    if (s1 == "")   return s2;
    if (s2 == "")   return s1;
    string maxx = s1, minn = s2;
    if (s1.length() < s2.length()){
        maxx = s2;
        minn = s1;
    }
    int a = maxx.length() - 1, b = minn.length() - 1;
    for (int i = b; i >= 0; --i){
        maxx[a--] += minn[i] - '0'; // a一直在减 ， 额外还要减个'0'
    }
    for (int i = maxx.length()-1; i > 0;--i){
        if (maxx[i] > '9'){
            maxx[i] -= 10;//注意这个是减10
            maxx[i - 1]++;
        }
    }
    if (maxx[0] > '9'){
        maxx[0] -= 10;
        maxx = '1' + maxx;
    }
    return maxx;
}
```

#### 1.3.2. 大数阶乘

```cpp
/*
    |大数模拟阶乘|
|用数组模拟|
*/
typedef long long LL;

const int maxn = 100010;
int num[maxn], len;
/*
    在mult函数中，形参部分：len每次调用函数都会发生改变，n表示每次要乘以的数，最终返回的是结果的长度
    tip: 阶乘都是先求之前的(n-1)!来求n!
    初始化Init函数很重要，不要落下
*/

void Init() {
    len = 1;
    num[0] = 1;
}

int mult(int num[], int len, int n) {
    LL tmp = 0;
    for(LL i = 0; i < len; ++i) {
         tmp = tmp + num[i] * n;    //从最低位开始，等号左边的tmp表示当前位，右边的tmp表示进位（之前进的位）
         num[i] = tmp % 10; // 保存在对应的数组位置，即去掉进位后的一位数
         tmp = tmp / 10;    // 取整用于再次循环,与n和下一个位置的乘积相加
    }
    while(tmp) {    // 之后的进位处理
         num[len++] = tmp % 10;
         tmp = tmp / 10;
    }
    return len;
}

int main() {
    Init();
    int n;
    n = 1977; // 求的阶乘数
    for(int i = 2; i <= n; ++i) {
        len = mult(num, len, i);
    }
    for(int i = len - 1; i >= 0; --i)
        printf("%d",num[i]);    // 从最高位依次输出,数据比较多采用printf输出
    printf("\n");
    return 0;
}
```

### 1.4. 最大公约数（GCD）

```cpp
/*
    |辗转相除法|
    |欧几里得算法|
	|求最大公约数|
*/
int gcd(int big, int small)
{
    if (small > big) 
        swap(big, small);
    int temp;
    while (small != 0){ // 辗转相除法
        if (small > big) 
            swap(big, small);
        temp = big % small;
        big = small;
        small = temp;
    }
    return(big);
}
```

### 1.5. 最小公倍数（LCM）

```cpp
int lcm (int big, int small) 
{
	return big * small / gcd(big, small);
}
```

### 1.6. 全排列

```cpp
/*
    |求1到n的全排列, 有条件|
*/
void Pern(int list[], int k, int n) {   // k表示前k个数不动仅移动后面n-k位数
    if (k == n - 1) {
        for (int i = 0; i < n; i++) {
            printf("%d", list[i]);
        }
        printf("\n");
	}
	else {
        for (int i = k; i < n; i++) {   // 输出的是满足移动条件所有全排列
            swap(list[k], list[i]);
            Pern(list, k + 1, n);
            swap(list[k], list[i]);
        }
    }
}
```

### 1.7. 二分搜索

```cpp
/*
    |二分搜索|
    |要求：先排序|
*/
// left为最开始元素, right是末尾元素的下一个数，x是要找的数
int bsearch(int *A, int left, int right, int x){
    int m;
    while (left < right){
        m = left + (right - left) / 2;
        if (A[m] >= x)  
            right = m;   
        else 
            left = m + 1;    
        // 如果要替换为 upper_bound, 改为 if (A[m] <= v) x = m+1; else y = m;     
    }
    return left;
}
/*
    最后left == right  
    如果找有多少的x，可以用lower_bound查找一遍，upper_bound查找一遍，下标相减。 
    cpp自带的lower_bound(a,a+n,x)返回数组中第一个x的地址, upper_bound(a,a+n,x)返回数组中最后一个x的下一个数的地址。如果a+n内没有找到x或x的下一个地址，返回a+n的地址  
*/
```



## 2. 数据结构

### 2.1. 并查集

```cpp
/*
    |合并节点操作|
*/
int father[maxn];   // 储存i的father父节点  

void makeSet() {  
    for (int i = 0; i < maxn; i++)   
        father[i] = i;  
}  

int findRoot(int x) {   // 迭代找根节点
    int root = x; // 根节点  
    while (root != father[root]) { // 寻找根节点  
        root = father[root];  
    }  
    while (x != root) {  
        int tmp = father[x];  
        father[x] = root; // 根节点赋值  
        x = tmp;  
    }  
    return root;  
}  

int findRoot(int x) {   // 迭代找根节点
	if (x == father[x])
		return x;
	else
		return father[x] = findRoot(father[x]);
}  

void Union(int x, int y) {  // 将x所在的集合和y所在的集合整合起来形成一个集合。  
    int a, b;  
    a = findRoot(x);  
    b = findRoot(y);  
    father[a] = b;  // y连在x的根节点上   或father[b] = a为x连在y的根节点上；  
} 
```

### 2.2. 最小生成树

#### 2.2.1. Kruskal算法

```cpp
/*
    |Kruskal算法|
    |适用于稀疏图|
*/
/*
    第一步：点、边、加入vector，把所有边按从小到大排序
    第二步：并查集部分 + 下面的code
*/
void Kruskal() {    
    ans = 0;    
    for (int i = 0; i < len; i++) {    
        if (Find(edge[i].a) != Find(edge[i].b)) {    
            Union(edge[i].a, edge[i].b);    
            ans += edge[i].len;    
        }    
    }    
} 
```

#### 2.2.2. Prim算法

```cpp
/*
    |Prim算法|
    |适用于稠密图|
    |堆优化版，时间复杂度：O(elgn)|
*/
//优先队列自定义比较函数
struct cmp {
	bool operator()(int &a, int &b) const {
		return a < b;
	}
};

struct node {  
    int v, len;  
    node(int v = 0, int len = 0) :v(v), len(len) {}  
    bool operator < (const node &a) const {  // 加入队列的元素自动按距离从小到大排序  
        return len > a.len;  
    }  
};

vector<node> G[maxn];
int vis[maxn];
int dis[maxn];

void init() {  
    for (int i = 0; i < maxn; i++) {  
        G[i].clear();  
        dis[i] = INF;  
        vis[i] = false;  
    }  
} 

int Prim(int s) {  
    priority_queue<node> Q; // 定义优先队列  
    int ans = 0;  
    Q.push(node(s,0));  // 起点加入队列  
    while (!Q.empty()) {   
        node now = Q.top(); 
		Q.pop();  // 取出距离最小的点  
        int v = now.v;  
        if (vis[v]) continue;  // 同一个节点，可能会推入2次或2次以上队列，这样第一个被标记后，剩下的需要直接跳过。  
        vis[v] = true;  // 标记一下  
        ans += now.len;  
        for (int i = 0; i < G[v].size(); i++) {  // 开始更新  
            int v2 = G[v][i].v;  
            int len = G[v][i].len;  
            if (!vis[v2] && dis[v2] > len) {   
                dis[v2] = len;  
                Q.push(node(v2, dis[v2]));  // 更新的点加入队列并排序  
            }  
        }  
    }  
    return ans; 
}
```

### 2.3. 单源最短路径

#### 2.3.1. Dijkstra算法

```cpp
/*
    |Dijkstra算法|
    |适用于边权为正的有向图或者无向图|
    |求从单个源点出发，到所有节点的最短路|
    |优化版：时间复杂度 O(elbn)|
*/
struct node {  
    int v, len;  
    node(int v = 0, int len = 0) :v(v), len(len) {}  
    bool operator < (const node &a) const {  // 距离从小到大排序  
        return len > a.len;  
    }  
};  

vector<node>G[maxn];  
bool vis[maxn];  
int dis[maxn];

void init() {  
    for (int i = 0; i<maxn; i++) {  
        G[i].clear();  
        vis[i] = false;  
        dis[i] = INF;  
    }  
} 

int dijkstra(int s, int e) {  
    priority_queue<node> Q;  
    Q.push(node(s, 0)); // 加入队列并排序  
    dis[s] = 0;  
    while (!Q.empty()) {  
        node now = Q.top();     // 取出当前最小的  
        Q.pop();  
        int v = now.v;  
        if (vis[v]) continue;   // 如果标记过了, 直接continue  
        vis[v] = true;  
        for (int i = 0; i < G[v].size(); i++) {   // 更新  
            int v2 = G[v][i].v;  
            int len = G[v][i].len;  
            if (!vis[v2] && dis[v2] > dis[v] + len) {  
                dis[v2] = dis[v] + len;  
                Q.push(node(v2, dis[v2]));  
            }  
        }  
    }  
    return dis[e];  
} 
```

#### 2.3.2. SPFA算法

```cpp
// 最短路径快速算法（Shortest Path Faster Algorithm）
/*
    |SPFA算法|
    |队列优化|
    |可处理负环|
*/
vector<node> G[maxn];
bool inqueue[maxn];
int dist[maxn];

void Init()  
{  
    for(int i = 0; i < maxn; ++i){  
        G[i].clear();  
        dist[i] = INF;  
    }  
}
  
int SPFA(int s,int e)  
{  
    int v1, v2, weight;  
    queue<int> Q;  
    memset(inqueue, false, sizeof(inqueue)); // 标记是否在队列中  
    memset(cnt, 0, sizeof(cnt)); // 加入队列的次数  
    dist[s] = 0;  
    Q.push(s); // 起点加入队列  
    inqueue[s] = true; // 标记  
    while(!Q.empty()){  
        v1 = Q.front();  
        Q.pop();  
        inqueue[v1] = false; // 取消标记  
        for(int i = 0; i < G[v1].size(); ++i){ // 搜索v1的链表  
            v2 = G[v1][i].vex;  
            weight = G[v1][i].weight;  
            if(dist[v2] > dist[v1] + weight){ // 松弛操作  
                dist[v2] = dist[v1] + weight;  
                if(inqueue[v2] == false){  // 再次加入队列  
                    inqueue[v2] = true;  
                    //cnt[v2]++;  // 判负环  
                    //if(cnt[v2] > n) return -1;  
                    Q.push(v2);  
                } } }  
    }  
    return dist[e];  
}
/*
    不断的将s的邻接点加入队列，取出不断的进行松弛操作，直到队列为空  
    如果一个结点被加入队列超过n-1次，那么显然图中有负环  
*/
```

#### 2.3.3. Floyd算法

```cpp
/*
    |Floyd算法|
    |任意点对最短路算法|
    |求图中任意两点的最短距离的算法|
*/
for (int i = 0; i < n; i++) {   // 初始化为0  
    for (int j = 0; j < n; j++)  
        scanf("%lf", &dis[i][j]);  
}

for (int k = 0; k < n; k++) {  
    for (int i = 0; i < n; i++) {  
        for (int j = 0; j < n; j++) {  
            dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);  
        }  
    }
}
```

### 2.4. 二分图

#### 2.4.1. 染色法

 ```cpp
/*
    |交叉染色法判断二分图|
*/
int bipartite(int s) {  
    int u, v;  
    queue<int> Q;  
    color[s] = 1;  
    Q.push(s);  
    while (!Q.empty()) {  
        u = Q.front();  
        Q.pop();  
        for (int i = 0; i < G[u].size(); i++) {  
            v = G[u][i];  
            if (color[v] == 0) {  
                color[v] = -color[u];  
                Q.push(v);  
            }  
            else if (color[v] == color[u])  
                return 0;  
        }  
    }  
    return 1;  
} 
 ```

#### 2.4.2. 匈牙利算法

```cpp
/*
    |求解最大匹配问题|
    |递归实现|
*/
vector<int> G[maxn];  
bool inpath[maxn];  // 标记  
int match[maxn];    // 记录匹配对象 

void init() {  
    memset(match, -1, sizeof(match));  
    for (int i = 0; i < maxn; ++i) {  
        G[i].clear();  
    }  
}

bool findpath(int k) {  
    for (int i = 0; i < G[k].size(); ++i) {  
        int v = G[k][i];  
        if (!inpath[v]) {  
            inpath[v] = true;  
            if (match[v] == -1 || findpath(match[v])) { // 递归  
                match[v] = k; // 即匹配对象是“k妹子”的  
                return true;  
            }  
        }  
    }  
    return false;  
}  

void hungary() {  
    int cnt = 0;  
    for (int i = 1; i <= m; i++) {  // m为需要匹配的“妹子”数  
        memset(inpath, false, sizeof(inpath)); // 每次都要初始化  
        if (findpath(i)) cnt++;  
    }  
    cout << cnt << endl;  
}
```

```cpp
/*
    |求解最大匹配问题|
    |dfs实现|
*/
int v1, v2;  
bool Map[501][501];  
bool visit[501];  
int link[501];  
int result;  

bool dfs(int x) {  
    for (int y = 1; y <= v2; ++y) {  
        if (Map[x][y] && !visit[y]) {  
            visit[y] = true;  
            if (link[y] == 0 || dfs(link[y])) {  
                link[y] = x;  
                return true;  
            } 
		} 
	}  
    return false;  
}  

void Search()  {  
    for (int x = 1; x <= v1; x++) {  
        memset(visit, false, sizeof(visit));  
        if (dfs(x))  
            result++;  
    }
}
```



## 3. 动态规划

### 3.1. 背包问题

```cpp
/*
    |01背包|
    |完全背包|
	|多重背包|
*/
// 01背包：  
void bag01(int cost, int weight) {  
    for(i = v; i >= cost; --i)  
    dp[i] = max(dp[i], dp[i - cost] + weight);  
}  

// 完全背包：  
void complete(int cost, int weight) {  
    for(i = cost; i <= v; ++i)  
    dp[i] = max(dp[i], dp[i - cost] + weight);  
}  

// 多重背包：  
void multiply(int cost, int weight, int amount) {  
    if(cost * amount >= v)  
        complete(cost, weight);  
    else{  
        k = 1;  
        while (k < amount){  
            bag01(k * cost, k * weight);  
            amount -= k;  
            k += k;  
        }  
        bag01(cost * amount, weight * amount);  
    }  
}  
```

### 3.2. 最长上升子序列（LIS）

```cpp
/*
    |最长上升子序列|
    |状态转移|
*/
/*
    状态转移dp[i] = max{1, dp[j] + 1 };  j<i; a[j]<a[i];
    d[i]是以i结尾的最长上升子序列
    与i之前的 每个a[j]<a[i]的 j的位置的最长上升子序列+1后的值比较
*/
void solve() {   // 参考挑战程序设计入门经典;
    for(int i = 0; i < n; ++i){  
        dp[i] = 1;  
        for(int j = 0; j < i; ++j){  
            if(a[j] < a[i]){  
                dp[i] = max(dp[i], dp[j] + 1);  
			}
		} 
	}
}  

/* 
    优化方法：
    dp[i]表示长度为i+1的上升子序列的最末尾元素  
    找到第一个比dp末尾大的来代替 
*/
void solve() {  
    for (int i = 0; i < n; ++i){
        dp[i] = INF;
    }
    for (int i = 0; i < n; ++i) {  
        *lower_bound(dp, dp + n, a[i]) = a[i];  // 返回一个指针  
    }  
    printf("%d\n", *lower_bound(dp, dp + n, INF) - dp;  
}
/*  
    函数lower_bound()返回一个 iterator 它指向在[first,last)标记的有序序列中可以插入value，而不会破坏容器顺序的第一个位置，而这个位置标记了一个不小于value的值。
*/
```

### 3.3. 最长公共子序列（LCS）

```cpp
/*
    |求最长公共子序列|
    |递推形式|
*/
void solve() {  
    for (int i = 0; i < n; ++i) {  
        for (int j = 0; j < m; ++j) {  
            if (s1[i] == s2[j]) {  
                dp[i + 1][j + 1] = dp[i][j] + 1;  
            }
            else {  
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);  
            } 
        } 
    }
}  
```



## 4. 字符串

### 4.1. kmp算法

```cpp
/*
    |kmp算法|
    |字符串匹配|
*/
void getnext(char str[maxn], int nextt[maxn]) {
    int j = 0, k = -1;
    nextt[0] = -1;
    while (j < m) {
        if (k == -1 || str[j] == str[k]) {
            j++;
            k++;
            nextt[j] = k;
        }
        else
            k = nextt[k];
    }
}

void kmp(int a[maxn], int b[maxn]) {    
    int nextt[maxm];    
    int i = 0, j = 0;    
    getnext(b, nextt);    
    while (i < n) {    
        if (j == -1 || a[i] == b[j]) { // 母串不动，子串移动    
            j++;    
            i++;    
        }    
        else {    
            // i不需要回溯了    
            // i = i - j + 1;    
            j = nextt[j];    
        }    
        if (j == m) {    
            printf("%d\n", i - m + 1); // 母串的位置减去子串的长度+1    
            return;    
        }    
    }    
    printf("-1\n");
}
```

### 4.2. kmp扩展算法

```cpp
const int MM = 100005;    

int next[MM], extend[MM];    
char S[MM], T[MM];    

void GetNext(const char *T) {    
    int len = strlen(T), a = 0;    
    next[0] = len;    
    while(a < len - 1 && T[a] == T[a + 1]) a++;    
    next[1] = a;    
    a = 1;    
    for(int k = 2; k < len; k++) {    
        int p = a + next[a] - 1, L = next[k - a];    
        if((k - 1) + L >= p) {    
            int j = (p - k + 1) > 0 ? (p - k + 1) : 0;    
            while(k + j < len && T[k + j] == T[j]) j++;    
            next[k] = j;    
            a = k;    
        }
        else 
            next[k] = L;    
    }    
} 

void GetExtend(const char *S, const char *T) {    
    GetNext(T);    
    int slen = strlen(S), tlen = strlen(T), a = 0;    
    int MinLen = slen < tlen ? slen : tlen;    
    while(a < MinLen && S[a] == T[a]) a++;    
    extend[0] = a;     
    a = 0;    
    for(int k = 1; k < slen; k++) {    
        int p = a + extend[a] - 1, L = next[k - a];    
        if((k - 1) + L >= p) {    
            int j = (p - k + 1) > 0 ? (p - k + 1) : 0;    
            while(k + j < slen && j < tlen && S[k + j] == T[j]) j++;    
            extend[k] = j;    
            a = k;    
        } 
        else    
            extend[k] = L;    
    }    
}    

void show(const int *s, int len){    
    for(int i = 0; i < len; i++)    
		cout << s[i] << ' ';    
    cout << endl;    
}    

int main() {    
    while(cin >> S >> T) {    
        GetExtend(S, T);    
        show(next, strlen(T));    
        show(extend, strlen(S));    
    }    
    return 0;    
}
```



## 5. 字典树

```cpp
struct Trie {  
    int cnt;  
    Trie *next[maxn];  
    Trie(){  
        cnt = 0;  
        memset(next,0,sizeof(next));  
    }  
};  

Trie *root;  

void Insert(char *word) {  
    Trie *tem = root;  
    while(*word != '\0') {  
        int x = *word - 'a';  
        if(tem->next[x] == NULL)  
            tem->next[x] = new Trie;  
        tem = tem->next[x];  
        tem->cnt++;  
        word++;  
    }  
}  

int Search(char *word) {  
    Trie *tem = root;  
    for(int i = 0; word[i] != '\0'; i++) {  
        int x = word[i]-'a';  
        if(tem->next[x] == NULL)  
            return 0;  
        tem = tem->next[x];  
    }  
    return tem->cnt;  
}  

void Delete(char *word, int t) {  
    Trie *tem = root;  
    for(int i = 0; word[i] != '\0'; i++) {  
        int x = word[i]-'a';  
        tem = tem->next[x];  
        (tem->cnt) -= t;  
    }  
    for(int i = 0; i < maxn; i++)  
        tem->next[i] = NULL;  
}  

int main() {  
    int n;  
    char str1[50];  
    char str2[50];  
    while(scanf("%d", &n)!=EOF) {  
        root = new Trie;  
        while(n--) {  
            scanf("%s %s", str1, str2);  
            if(str1[0] == 'i') {
                Insert(str2); 
            }
            else if(str1[0] == 's') {  
                if(Search(str2))  
                    printf("Yes\n");  
                else  
                    printf("No\n");  
            }
            else {  
                int t = Search(str2);  
                if(t)  
                    Delete(str2, t);  
            } 
        } 
    }  
    return 0;  
}  
```



## 6. 线段树

### 6.1. 点更新

```cpp
struct node
{
    int left, right;
    int max, sum;
};

node tree[maxn << 2];
int a[maxn];
int n;
int k = 1;
int p, q;
string str;

void build(int m, int l, int r)//m 是 树的标号
{
    tree[m].left = l;
    tree[m].right = r;
    if (l == r) {
        tree[m].max = a[l];
        tree[m].sum = a[l];
        return;
    }
    int mid = (l + r) >> 1;
    build(m << 1, l, mid);
    build(m << 1 | 1, mid + 1, r);
    tree[m].max = max(tree[m << 1].max, tree[m << 1 | 1].max);
    tree[m].sum = tree[m << 1].sum + tree[m << 1 | 1].sum;
}

void update(int m, int a, int val)//a 是 节点位置， val 是 更新的值（加减的值）
{
    if (tree[m].left == a && tree[m].right == a) {
        tree[m].max += val;
        tree[m].sum += val;
        return;
    }
    int mid = (tree[m].left + tree[m].right) >> 1;
    if (a <= mid) {
        update(m << 1, a, val);
    }
    else {
        update(m << 1 | 1, a, val);
    }
    tree[m].max = max(tree[m << 1].max, tree[m << 1 | 1].max);
    tree[m].sum = tree[m << 1].sum + tree[m << 1 | 1].sum;
}

int querySum(int m, int l, int r)
{
    if (l == tree[m].left && r == tree[m].right) {
        return tree[m].sum;
    }
    int mid = (tree[m].left + tree[m].right) >> 1;
    if (r <= mid) {
        return querySum(m << 1, l, r);
    }
    else if (l > mid) {
        return querySum(m << 1 | 1, l, r);
    }
    return querySum(m << 1, l, mid) + querySum(m << 1 | 1, mid + 1, r);
}

int queryMax(int m, int l, int r)
{
    if (l == tree[m].left && r == tree[m].right) {
        return tree[m].max;
    }
    int mid = (tree[m].left + tree[m].right) >> 1;
    if (r <= mid) {
        return queryMax(m << 1, l, r);
    }
    else if (l > mid) {
        return queryMax(m << 1 | 1, l, r);
    }
    return max(queryMax(m << 1, l, mid), queryMax(m << 1 | 1, mid + 1, r));
} 

build(1, 1, n);  
update(1, a, b);  
query(1, a, b); 
```

### 6.2. 区间更新

```cpp
typedef long long ll;  
const int maxn = 100010;  

int t,n,q;  
ll anssum;  

struct node{  
    ll l, r;  
    ll addv, sum;  
}tree[maxn << 2];  

void maintain(int id) {  
    if(tree[id].l >= tree[id].r)  
        return;  
    tree[id].sum = tree[id << 1].sum + tree[id << 1 | 1].sum;  
}  

void pushdown(int id) {  
    if(tree[id].l >= tree[id].r)  
        return;  
    if(tree[id].addv){  
        int tmp = tree[id].addv;  
        tree[id << 1].addv += tmp;  
        tree[id << 1 | 1].addv += tmp;  
        tree[id << 1].sum += (tree[id << 1].r - tree[id << 1].l + 1) * tmp;  
        tree[id << 1 | 1].sum += (tree[id << 1 | 1].r - tree[id << 1 | 1].l + 1) * tmp;  
        tree[id].addv = 0;  
    }  
}  

void build(int id, ll l, ll r) {  
    tree[id].l = l;  
    tree[id].r = r;  
    tree[id].addv = 0;  
    tree[id].sum = 0;  
    if(l == r) {  
        tree[id].sum = 0;  
        return;  
    }  
    ll mid = (l + r) >> 1;  
    build(id << 1, l, mid);  
    build(id << 1 | 1, mid + 1, r);  
    maintain(id);  
}  

void updateAdd(int id,ll l,ll r,ll val) {  
    if(tree[id].l >= l && tree[id].r <= r)  
    {  
        tree[id].addv += val;  
        tree[id].sum += (tree[id].r - tree[id].l + 1) * val;  
        return;  
    }  
    pushdown(id);  
    ll mid = (tree[id].l + tree[id].r) >> 1;  
    if(l <= mid)  
        updateAdd(id << 1, l, r, val);  
    if(mid < r)  
        updateAdd(id << 1 | 1, l, r, val);  
    maintain(id);  
}  

void query(int id, ll l, ll r) {  
    if(tree[id].l >= l && tree[id].r <= r){  
        anssum += tree[id].sum;  
        return;  
    }  
    pushdown(id);  
    ll mid = (tree[id].l + tree[id].r) >> 1;  
    if(l <= mid)  
        query(id << 1, l, r);  
    if(mid < r)  
        query(id << 1 | 1, l, r);  
    maintain(id);  
}  

int main() {  
    scanf("%d", &t);  
    int kase = 0;  
    while(t--) {  
        scanf("%d %d", &n, &q);  
        build(1, 1, n);  
        int id;  
        ll x, y;  
        ll val;  
        printf("Case %d:\n", ++kase);  
        while(q--) {  
            scanf("%d", &id);  
            if(id == 0) {  
                scanf("%lld %lld %lld", &x, &y, &val);  
                updateAdd(1, x + 1, y + 1, val);  
            }  
            else {  
                scanf("%lld %lld", &x, &y);  
                anssum = 0;  
                query(1, x + 1, y + 1);  
                printf("%lld\n", anssum);  
            } 
        } 
    }  
    return 0;  
}  
```



## 7. 树状数组

```cpp
typedef long long ll;

const int maxn = 50005;

int a[maxn];
int n;

int lowbit(const int t) {
    return t & (-t);
}

void insert(int t, int d) {
    while (t <= n){
        a[t] += d;
        t = t + lowbit(t);
    }
}

ll getSum(int t) {
    ll sum = 0;
    while (t > 0){
        sum += a[t];
        t = t - lowbit(t);
    }
    return sum;
}

int main() {
    int t, k, d;
    scanf("%d", &t);
    k= 1;
    while (t--){
        memset(a, 0, sizeof(a));
        scanf("%d", &n);
        for (int i = 1; i <= n; ++i) {
            scanf("%d", &d);
            insert(i, d);
        }
        string str;
        printf("Case %d:\n", k++);
        while (cin >> str) {
            if (str == "End")   break;
            int x, y;
            scanf("%d %d", &x, &y);
            if (str == "Query")
                printf("%lld\n", getSum(y) - getSum(x - 1));
            else if (str == "Add")
                insert(x, y);
            else if (str == "Sub")
                insert(x, -y);
        }
    }
    return 0;
}
```

```cpp
// 求逆序对
for(int i = 1; i <= n; i++)
	{
		scanf("%d", &a);
		node[i].index = i;
		node[i].v = a;
	}
	sort(node + 1, node + 1 + n);
	long long ans=0;
	for(int i = 1; i <= n; i++)
	{ 
		add(node[i].index);  //离散化结果—— 下标等效于数值
		ans += i - sum(node[i].index); //得到之前有多少个比你大的数（逆序对）
	}
	cout << ans;
```



## 8. 中国剩余定理（孙子定理）

```cpp
int CRT(int a[], int m[], int n)  {    
    int M = 1;    
    int ans = 0;    
    for(int i = 1; i <= n; i++)    
        M *= m[i];    
    for(int i = 1; i <= n; i++)  {    
        int x, y;    
        int Mi = M / m[i];    
        extend_Euclid(Mi, m[i], x, y);    
        ans = (ans + Mi * x * a[i]) % M;    
    }    
    if(ans < 0) ans += M;    
    return ans;    
}  

void extend_Euclid(int a, int b, int &x, int &y)  {  
    if(b == 0) {  
        x = 1;  
        y = 0;  
        return;  
    }  
    extend_Euclid(b, a % b, x, y);  
    int tmp = x;  
    x = y;  
    y = tmp - (a / b) * y;  
}  
```



## 9. 最大流/最大权闭合子图

```cpp
//从源点s向每个正权点连一条容量为权值的边，每个负权点向汇点t连一条容量为权值的绝对值的边，有向图原来的边容量全部为无限大。
//最大权闭合子图=（正权之和-不选的正权之和-要选的负权绝对值之和）=正权值和-最小割/最大流
#define maxn 5010     //课程
#define maxm 50100    //用户
#define inf 0x3f3f3f3f  
using namespace std;  

struct Edge {  
    int v, c, next;  
    Edge(int v, int c, int next): v(v), c(c), next(next) {}  
    Edge(){}  
}e[maxm * 6 + maxn * 2];  
int p[maxn + maxm];  
int cnt, n, m, T;  

void init() {  
    cnt = 0;  
    memset(p, -1, sizeof(p));  
} 

void insert(int u, int v, int c) {  
    e[cnt] = Edge(v, c, p[u]);  
    p[u] = cnt++;  // 顶点u的上一条相邻的边 
} 

int d[maxn + maxm];  
bool bfs() {  
    memset(d, -1, sizeof(d));  
    queue<int> q;  
    d[0] = 0;  
    q.push(0);  
    while(!q.empty()) {  
        int u = q.front();
        q.pop();  
        for(int i = p[u]; i != -1; i = e[i].next) {  
            int v = e[i].v;  
            if(e[i].c > 0 && d[v] == -1){  
                //printf("%d->%d(%d)\n", u, v, d[u] + 1);  
                d[v] = d[u] + 1;  
                q.push(v);  
            }  
        }  
    }  
    return d[T] != -1;  
} 

int dfs(int u, int flow){  
    if(u == T)
        return flow;  
    int res = 0;  
    for(int i = p[u]; i != -1; i = e[i].next){  
        int v = e[i].v;  
        if(e[i].c > 0 && d[v] == d[u] + 1){  
            int tmp = dfs(v, min(flow, e[i].c));  
            e[i].c -= tmp;  
            flow -= tmp;  
            e[i^1].c += tmp;  
            res += tmp;  
            if(flow == 0)  
                break;  
        }  
    }  
    if(res == 0)  
        d[u] = -1;  
    return res;  
}  

int dinic() {  
    int res = 0;  
    while(bfs()){  
   // printf("here!\n");  
        res += dfs(0, inf);  
    }  
    return res;  
}  

int main() {  
    init();  
    int p, a, b, c, sum = 0;  
    scanf("%d%d", &n, &m);  
    T = n + m + 1;//汇点  
    for(int i = 1; i <= n; i++){  
        scanf("%d", &p);  
        insert(i + m, T, p);  //课程放右边
        insert(T, i + m, 0);  
    }  
    for(int i = 1; i <= m; i++){  
        scanf("%d%d%d", &a, &b, &c);  
        sum += c;  
        insert(i, a + m, inf);   
        insert(a + m, i, 0);  
        insert(i, b + m, inf);  
        insert(b + m, i, 0);  
        insert(0, i, c);   //用户放左边
        insert(i, 0, 0);  
    }  
    printf("%d\n", sum - dinic());  
    return 0;  
```



## 10. 拓扑排序/AOE网络/关键路径

```cpp
const int maxn = 110;
const int INF = 1e4;
int N,M;
struct Node{
	//vector<int> child;
	int id; 
	int length;
};
//Node graph[maxn]; 
vector<Node> Adj[maxn];
//int e[maxn]; //边上活动最早开始时间 
//int l[maxn]; //边上活动最晚开始时间
int ve[maxn];   //顶点上活动最早开始时间 
int vl[maxn];  //顶点上活动最晚开始时间
int in[maxn];  // 每个结点的入度，为0时入队
stack<int> s;

bool TopologicalSort(int N)
{
	queue<int> q;
	memset(ve, 0, sizeof(ve));
	//memset(inq,0,sizeof(inq));
	/*先找出所有初始时入度为0的结点*/
	for(int i = 1; i <= N; i++)  
	{
		if(in[i] == 0)
		{
			q.push(i);
			//s.push(i);
			//inq[i] = true;
			//ve[i] = 0; 
		}
	}
	/*每次将所有入度为0的结点入栈，拓扑序*/
	while(!q.empty())
	{
		int tmp = q.front();
		q.pop();    
		s.push(tmp); // num++；
		//cout << "tmp:" << tmp << endl;
		for(int i = 0; i < Adj[tmp].size(); i++)
		{
			int id = Adj[tmp][i].id;
			if(--in[id] == 0) //入度减为0 加入拓扑排序 
			{
				q.push(id);
				//s.push(i);
				//inq[i]=true;
			}
			if(ve[tmp] + Adj[tmp][i].length > ve[id]) //更新ve值 
				ve[id] = ve[tmp] + Adj[tmp][i].length;
			
		}
	 } 
	 //cout << "size: " << s.size() << endl;
	 if(s.size() == N) 
		return true;
	 else 
		return false;
} 

void calc_path(int N)
{
	if(TopologicalSort(N) == false)
	{
		printf("0\n");
		return;
	}
	 /*寻找拓扑序列最后一个结点，即开始时间最晚的一个结点*/
	 int max = -1, u = -1;
	 for(int i = 1; i <= N; i++)
	 {
	 	if(ve[i] > max)
	   {
	    	max = ve[i];
	    	u = i;
	   } 
	 }
	  //fill(vl, vl + maxn, INF); 
	  //vl[u] = ve[u];
	  fill(vl, vl + maxn, ve[u]); 
	  printf("%d\n", ve[u]);
	  
	 /*元素逐个出栈，即为逆拓扑序列，构造vl数组*/
	 while(!s.empty())
	 {
	 	int tmp = s.top();
	 	s.pop();
	 	//int min = INF, u;
	 	for(int i = 0; i < Adj[tmp].size(); i++)
	 	{
	 		int id = Adj[tmp][i].id;
	 		if(vl[id] - Adj[tmp][i].length < vl[tmp])
	 		{
	 			vl[tmp] = vl[id] - Adj[tmp][i].length;
			 }
		 }
	 }
	 /*遍历邻接点每条边，计算每项活动的最早和最晚开始时间*/
	 for(int i = 1; i <= N; i++)
	 {
	 	for(int j = Adj[i].size() - 1; j >= 0; j--)
	 	{
	 		int id = Adj[i][j].id;
	 		int e = ve[i];
	 		int l = vl[id]-Adj[i][j].length;
	 		if(e == l) 
				printf("%d->%d\n", i, id); 
		 }
	 }
 } 
 
int main()
{
	while(scanf("%d%d", &N, &M) != EOF)
	{
		int v, w, len;
		for(int i = 1; i <= N; i++)
		Adj[i].clear();
		memset(in, 0, sizeof(in));
		for(int i = 0; i < M; i++)
		{
			scanf("%d%d%d", &v, &w, &len);
			Node tmp;
			tmp.id = w;
			tmp.length = len;
			Adj[v].push_back(tmp); //有向图只要添加单向边即可 
			in[w]++; 
		}
		while(!s.empty())
		s.pop();
		calc_path(N);
		//cout << "end!!" << endl;
	}
	return 0;
}
```



## 11. 强连通分量

```cpp
const int N=100010;
struct data
{
    int to, next;
} tu[N * 2];
int head[N];
int ip;
int dfn[N], low[N];///dfn[]表示深搜的步数，low[u]表示u或u的子树能够追溯到的最早的栈中节点的次序号
int sccno[N];///缩点数组，表示某个点对应的缩点值
int step;
int scc_cnt;///强连通分量个数

void init()
{
    ip=0;
    memset(head, -1, sizeof(head));
}

void add(int u, int v)
{
    tu[ip].to = v;
    tu[ip].next = head[u];
    head[u] = ip++;
}

vector<int> scc[N];///得出来的缩点，scc[i]里面存i这个缩点具体缩了哪些点
stack<int> S;
void dfs(int u)
{
    dfn[u] = low[u] = ++step;
    S.push(u);
    for (int i = head[u]; i != -1; i = tu[i].next)
    {
        int v = tu[i].to;
        if (!dfn[v])
        {
            dfs(v);
            low[u] = min(low[u], low[v]);
        }
        else if (!sccno[v])
            low[u] = min(low[u], dfn[v]);
    }
    if (low[u] == dfn[u])
    {
        scc_cnt += 1;
        scc[scc_cnt].clear();
        while(1)
        {
            int x = S.top();
            S.pop();
            if (sccno[x] != scc_cnt) 
				scc[scc_cnt].push_back(x);
            sccno[x] = scc_cnt;
            if (x == u) 
				break;
        }
    }
}

void tarjan(int n)
{
    memset(sccno, 0, sizeof(sccno));
    memset(dfn, 0, sizeof(dfn));
    step = scc_cnt = 0;
    for (int i = 1; i <= n; i++)
        if (!dfn[i]) 
			dfs(i);
}
```



## 12. 日期与星期/蔡勒公式

```cpp
int getDayofWeek(int y, int m, int d){
    if(m == 1 || m == 2) {
        m += 12;
        y--;
    }
    return (d + 2 * m + 3 * (m + 1) / 5 + y + y / 4 - y / 100 + y / 400 + 1) % 7;
} 
```



## 参考

- [ACM算法模板](https://blog.csdn.net/qq_32265245/article/details/53046750)

- [dinic求最大权闭合子图](https://blog.csdn.net/m0_38033475/article/details/80173037)

- [拓扑排序+AOE网络+关键路径](https://blog.csdn.net/weixin_42584977/article/details/92001428)

- [tarjan模板](https://blog.csdn.net/martinue/article/details/51315574)

- [蔡勒公式](https://zh.wikipedia.org/wiki/%E8%94%A1%E5%8B%92%E5%85%AC%E5%BC%8F)
