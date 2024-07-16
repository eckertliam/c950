from typing import Generic, TypeVar, List, Tuple, Optional

"""Key-Value type variable"""
K = TypeVar('K')

"""Value type variable"""
V = TypeVar('V')


class Map(Generic[K, V]):
    """Map Class: a simple hash map implementation"""
    def __init__(self, size: int = 50) -> None:
        self.size = size
        self.buckets: List[List[Tuple[K, V]]] = [[] for _ in range(size)]

    def _hash(self, key: K) -> int:
        """Private hash function to determine the bucket index for a key"""
        return hash(key) % self.size

    def set(self, key: K, value: V) -> None:
        """
        Set a key value pair into the map or update the value if key already exists.
        Best case O(1) if key is at the beginning of the bucket or bucket is empty.
        Worst case O(N) if key is at the end of the bucket.
        """
        idx = self._hash(key)
        # check if bucket is empty
        if not self.buckets[idx]:
            self.buckets[idx].append((key, value))
            return
        # check if key already exists
        for i, (k, _) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        # key does not exist
        self.buckets[idx].append((key, value))

    def get(self, key: K) -> Optional[V]:
        """Get a value by key.
        Best case O(1) if key is at the beginning of the bucket.
        Worst case O(N) if key is at the end of the bucket.
        """
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return None

    def __setitem__(self, key, value) -> None:
        self.set(key, value)

    def __getitem__(self, key) -> Optional[V]:
        return self.get(key)

    def list_all(self) -> List[Tuple[K, V]]:
        """List all key-value pairs in the map"""
        return [item for bucket in self.buckets for item in bucket]

    def __iter__(self):
        return iter(self.list_all())

    def list_keys(self) -> List[K]:
        """List all keys in the map"""
        return [k for k, _ in self.list_all()]

    def list_values(self) -> List[V]:
        """List all values in the map"""
        return [v for _, v in self.list_all()]
