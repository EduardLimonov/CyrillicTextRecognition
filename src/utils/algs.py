from collections import deque
from typing import *


def get_connect_components(adj_lists: Dict[int, List[int]]) -> List[Set[int]]:
    # имеем списки смежности, помещенные в словарь; ключ -- вершина, значение -- список смежных с ней вершин
    # возвращает список всех компонент связности в данном графе
    visited = set()
    comps = []

    for v in adj_lists.keys():
        if v in visited:
            continue
        new_comp = {v}
        visited.add(v)
        should_visit = deque(adj_lists[v])
        while len(should_visit):  # обойдем все смежные с v вершины
            u = should_visit.popleft()
            new_comp.add(u)
            visited.add(u)
            for t in adj_lists[u]:
                if t not in visited and t not in should_visit:
                    # мы еще не заходили в t ни при обходе компоненты, ни до этого
                    should_visit.append(t)

        comps.append(new_comp)

    return comps


def unite(components: List[Set[int]], to_unite: List[Tuple[int, int]]) -> List[Set[int]]:

    def unite_sets(sets: Iterable[Set]) -> Set:
        # объединяет все множества в наборе
        ans = set()
        for s in sets:
            ans = ans.union(s)

        return ans

    adj_list = dict()

    # для каждого номера компоненты определим все другие компоненты, с которыми мы будем ее объединять
    for u1, u2 in to_unite:
        for u in (u1, u2):
            if u not in adj_list.keys():
                adj_list[u] = []

        if u2 not in adj_list[u1]:
            adj_list[u1].append(u2)
        if u1 not in adj_list[u2]:
            adj_list[u2].append(u1)

    supercomponents: List[Set[int]] = get_connect_components(adj_list)  # список множеств; каждое множество --
    # номера компонент связности, которые нужно будет объединить между собой

    const_comp = [i for i in range(len(components)) if i not in adj_list.keys()]

    return [
        unite_sets(
            components[i] for i in supercomponent
        )
        for supercomponent in supercomponents
    ] + [components[j] for j in const_comp]
    # наконец-то объединяем множества пикселов; индексы объединяемых множеств содержатся в supercomponents
    # множества const_comp оставляем неизменными и просто добавляем в ответ

