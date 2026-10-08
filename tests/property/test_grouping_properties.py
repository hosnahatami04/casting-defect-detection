from hypothesis import given
from hypothesis import strategies as st

from castinspect.data.phash import connected_groups


def _bfs_reference(size: int, edges: list[tuple[int, int]]) -> list[int]:
    adjacency: dict[int, set[int]] = {i: set() for i in range(size)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    component = [-1] * size
    current = 0
    for start in range(size):
        if component[start] != -1:
            continue
        queue = [start]
        component[start] = current
        while queue:
            node = queue.pop()
            for neighbour in adjacency[node]:
                if component[neighbour] == -1:
                    component[neighbour] = current
                    queue.append(neighbour)
        current += 1
    return component


@st.composite
def _graphs(draw: st.DrawFn) -> tuple[int, list[tuple[int, int]]]:
    size = draw(st.integers(min_value=1, max_value=40))
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=size - 1),
                st.integers(min_value=0, max_value=size - 1),
            ),
            max_size=80,
        )
    )
    return size, edges


@given(_graphs())
def test_union_find_matches_bfs_reference(graph: tuple[int, list[tuple[int, int]]]) -> None:
    size, edges = graph
    assert connected_groups(size, edges) == _bfs_reference(size, edges)
