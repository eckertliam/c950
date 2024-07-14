from typing import TypeVar, Optional, List, Iterator, Tuple, Generic

"""key type"""
K = TypeVar('K')

"""value type"""
V = TypeVar('V')


class Map(Generic[K, V]):
    def __init__(self):
        self.keys: List[K] = []
        self.values: List[V] = []

    def push(self, key: K, value: V) -> None:
        self.keys.append(key)
        self.values.append(value)

    def get(self, key: K) -> Optional[V]:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i]
        return None

    def remove(self, key: K) -> None:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                self.keys.pop(i)
                self.values.pop(i)
                break

    def __contains__(self, key: K) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, key: K) -> Optional[V]:
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.push(key, value)
