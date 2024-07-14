from collections import deque

def bfs(graph, start):
    visited = []
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.append(vertex)
            queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)
    
    return visited

def dfs(graph, start):
    visited = []
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            stack.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)
    
    return visited

# Example usage
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
print(bfs(graph, 2))  # Output: [2, 0, 3, 1]
print(dfs(graph, 2))  # Output: [2, 3, 0, 1]
