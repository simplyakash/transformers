# 🧠 Time Complexity of Common Algorithms and Operations

---

# 📊 Complexity Cheat Sheet

| Complexity | Name | Example |
|---|---|---|
| $O(1)$ | Constant Time | Array indexing |
| $O(\log n)$ | Logarithmic | Binary Search |
| $O(n)$ | Linear | Linear Search |
| $O(n \log n)$ | Linearithmic | Merge Sort |
| $O(n^2)$ | Quadratic | Bubble Sort |
| $O(n^3)$ | Cubic | Floyd Warshall |
| $O(2^n)$ | Exponential | Recursive Fibonacci |
| $O(n!)$ | Factorial | Traveling Salesman Brute Force |

---

# 📌 1️⃣ Array Operations

| Operation | Time Complexity |
|---|---|
| Access by index | $O(1)$ |
| Update | $O(1)$ |
| Search | $O(n)$ |
| Insert at end | $O(1)$ |
| Insert at beginning | $O(n)$ |
| Delete at beginning | $O(n)$ |

---

# 📌 2️⃣ Linked List Operations

| Operation | Time Complexity |
|---|---|
| Access | $O(n)$ |
| Search | $O(n)$ |
| Insert at head | $O(1)$ |
| Insert at tail | $O(1)$ (if tail pointer exists) |
| Delete head | $O(1)$ |

---

# 📌 3️⃣ Stack Operations

| Operation | Time Complexity |
|---|---|
| Push | $O(1)$ |
| Pop | $O(1)$ |
| Peek | $O(1)$ |

---

# 📌 4️⃣ Queue Operations

| Operation | Time Complexity |
|---|---|
| Enqueue | $O(1)$ |
| Dequeue | $O(1)$ |
| Front | $O(1)$ |

---

# 📌 5️⃣ HashMap / Dictionary

| Operation | Average Case | Worst Case |
|---|---|---|
| Insert | $O(1)$ | $O(n)$ |
| Search | $O(1)$ | $O(n)$ |
| Delete | $O(1)$ | $O(n)$ |

---

# 📌 6️⃣ Binary Search Tree (BST)

| Operation | Average | Worst |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |

---

# 📌 7️⃣ Heap Operations

| Operation | Complexity |
|---|---|
| Insert | $O(\log n)$ |
| Delete Min/Max | $O(\log n)$ |
| Peek | $O(1)$ |

---

# 📌 8️⃣ Sorting Algorithms

| Algorithm | Best | Average | Worst |
|---|---|---|---|
| Bubble Sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ |
| Selection Sort | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ |
| Insertion Sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ |
| Merge Sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |
| Quick Sort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ |
| Heap Sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |

---

# 📌 9️⃣ Searching Algorithms

| Algorithm | Complexity |
|---|---|
| Linear Search | $O(n)$ |
| Binary Search | $O(\log n)$ |

---

# 📌 🔟 Graph Algorithms

| Algorithm | Complexity |
|---|---|
| BFS | $O(V + E)$ |
| DFS | $O(V + E)$ |
| Dijkstra | $O((V+E)\log V)$ |
| Bellman Ford | $O(VE)$ |
| Floyd Warshall | $O(V^3)$ |
| Kruskal | $O(E \log E)$ |
| Prim’s | $O(E \log V)$ |

Where:
- $V$ → vertices
- $E$ → edges

---

# 📌 1️⃣1️⃣ Dynamic Programming Examples

| Problem | Complexity |
|---|---|
| Fibonacci DP | $O(n)$ |
| Knapsack | $O(nW)$ |
| Longest Common Subsequence | $O(nm)$ |

---

# 📌 1️⃣2️⃣ Recursion Examples

| Problem | Complexity |
|---|---|
| Recursive Fibonacci | $O(2^n)$ |
| Tower of Hanoi | $O(2^n)$ |
| Subset Generation | $O(2^n)$ |
| Permutations | $O(n!)$ |

---

# 📊 Space Complexity Examples

| Algorithm | Space Complexity |
|---|---|
| Merge Sort | $O(n)$ |
| Quick Sort | $O(\log n)$ |
| BFS | $O(V)$ |
| DFS Recursive | $O(V)$ |

---

# 🧠 Complexity Growth Order

```text
O(1)
   ↓
O(log n)
   ↓
O(n)
   ↓
O(n log n)
   ↓
O(n²)
   ↓
O(2ⁿ)
   ↓
O(n!)
```

Lower is better.

---

# 📌 Important Interview Notes

| Complexity | Interview Interpretation |
|---|---|
| $O(1)$ | Excellent |
| $O(\log n)$ | Very efficient |
| $O(n)$ | Acceptable |
| $O(n \log n)$ | Ideal for sorting |
| $O(n^2)$ | Slow for large inputs |
| $O(2^n)$ | Usually impractical |

---

# 🎤 Interview-Friendly Explanation

> “Time complexity measures how the runtime of an algorithm grows with input size. Efficient algorithms generally aim for logarithmic or linear complexity, while quadratic and exponential complexities become expensive for large datasets.”
