"""
custom_pq.py - Custom priority queues for A* tie-breaking
"""

import heapq


class CustomPQ_maxG:
    """Priority queue that breaks ties in favor of larger g-values"""
    def __init__(self):
        self.heap = []

    def push(self, f, g, state):
        # Use -g so that among same f, larger g gets popped first
        heapq.heappush(self.heap, (f, -g, state))

    def pop(self):
        f, neg_g, state = heapq.heappop(self.heap)
        return f, -neg_g, state

    def is_empty(self):
        return len(self.heap) == 0


class CustomPQ_minG:
    """Priority queue that breaks ties in favor of smaller g-values"""
    def __init__(self):
        self.heap = []

    def push(self, f, g, state):
        # Use +g so that among same f, smaller g gets popped first
        heapq.heappush(self.heap, (f, g, state))

    def pop(self):
        f, g, state = heapq.heappop(self.heap)
        return f, g, state

    def is_empty(self):
        return len(self.heap) == 0