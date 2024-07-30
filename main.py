from typing import List, Optional, Tuple, TypeVar, Generic, Union
import csv
from dataclasses import dataclass, field
from datetime import time

# DATA STRUCTURES

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

    def insert(self, key: K, value: V) -> None:
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

    def remove(self, key: K) -> None:
        """Remove a key-value pair by key.
        Best case O(1) if key is at the beginning of the bucket.
        Worst case O(N) if key is at the end of the bucket.
        """
        idx = self._hash(key)
        for i, (k, _) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx].pop(i)
                return

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
        self.insert(key, value)

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

    def __len__(self):
        return len(self.list_all())


@dataclass
class Address:
    """Address is all human friendly address info address, zip, city
    this is stored in the address to id map raw address IDs are used everywhere else besides the UI
    because it is much easier and more efficient to work with integers than a set of strings"""
    address: str
    zip_code: str
    city: str
    address_id: int


class AddressMap:
    """AddressMap is a map of addresses to address ids and vice versa"""
    def __init__(self):
        self.addresses = Map[str, Address]
        self.ids = Map[int, Address]

    def insert(self, address: Address) -> None:
        """Insert an address into the map"""
        self.addresses.insert(address.address, address)
        self.ids.insert(address.address_id, address)

    def get_by_address(self, address: str) -> Address:
        """Get an address by address string"""
        return self.addresses.get(address)

    def get_by_id(self, address_id: int) -> Address:
        """Get an address by address id"""
        return self.ids.get(address_id)


"""Status type variable 'HUB' | 'EN_ROUTE' | 'DELIVERED'"""
Status = Union['HUB', 'EN_ROUTE', 'DELIVERED']


@dataclass
class Package:
    """Package points to an address, package weight in pounds, package id, delivery deadline
    status of the package, a required truck id if given, a truck id if assigned,
    a delay time if the package is delayed, and a list of package ids it must be delivered with if any"""
    address_id: int
    weight: float
    pack_id: int
    deadline: Optional[time] = None
    status: Status = 'HUB'
    requires_truck: Optional[int] = None
    truck_id: Optional[int] = None
    delay: Optional[time] = None
    requires_packs: List[int] = field(default_factory=list)


"""type definition of a DistanceMatrix as a list of lists of floats 
allows for efficient O(1) access to distances between addresses"""
DistanceMatrix = List[List[float]]


class PackageMap:
    """efficiently store and access packages by package id at O(1)
    no hash function needed, just use the package id as the index"""
    def __init__(self, size: int = 40):
        self.size = size
        self.packages: List[Optional[Package]] = [None] * size

    def _expand(self, new_size: int) -> None:
        self.packages += [None] * (new_size - self.size)
        self.size = new_size

    def insert(self, package: Package) -> None:
        # check if it is necessary to expand the list
        if self.size <= package.pack_id:
            self._expand(package.pack_id + 1)
        self.packages[package.pack_id] = package

    def get(self, pack_id: int) -> Optional[Package]:
        return self.packages[pack_id]

    def remove(self, pack_id: int) -> None:
        self.packages[pack_id] = None

    def mark_delivered(self, pack_id: int) -> None:
        """mark a package as delivered"""
        package = self.get(pack_id)
        if package:
            package.status = 'DELIVERED'
            self.insert(package)

    def mark_en_route(self, pack_id: int) -> None:
        """mark a package as en route"""
        package = self.get(pack_id)
        if package:
            package.status = 'EN_ROUTE'
            self.insert(package)

    def mark_packs_en_route(self, pack_ids: List[int]) -> None:
        """mark a list of packages as en route"""
        for pack_id in pack_ids:
            self.mark_en_route(pack_id)

    def add_address_to_route(self, address_id: int) -> List[int]:
        """get all packages at an address"""
        return [package.pack_id for package in self.packages if package and package.address_id == address_id]


@dataclass
class Truck:
    """Truck is a vehicle that delivers packages to addresses
    it has a truck id, a list of package ids it is carrying, and a list of addresses it has visited"""
    truck_id: int
    hasDriver: bool
    route: list[int] = field(default_factory=list)
    max_packs: int = 16
    packages: List[int] = field(default_factory=list)
    visited: List[int] = field(default_factory=list)

    def add_to_route(self, address_id: int, package_map: PackageMap) -> bool:
        """add an address to the truck route returns true if the truck has room for the address"""
        new_packs = package_map.add_address_to_route(address_id)
        if len(self.packages) + len(new_packs) > self.max_packs:
            return False
        self.packages += new_packs
        self.route.append(address_id)
        package_map.mark_packs_en_route(new_packs)
        return True

# UTILITY FUNCTIONS


def distance_matrix_from_csv(file_path: str) -> DistanceMatrix:
    """read a distance matrix from a csv file"""
    matrix: DistanceMatrix = []
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    next(reader)
    for row in reader:
        # trim the first element
        row = row[1:]
        new_row = [float(distance) for distance in row]
        matrix.append(new_row)
    return matrix


def read_address_row(row: List[str]) -> Address:
    """read an address from a csv row"""
    return Address(row[0], row[2], row[1], int(row[3]))


def read_address_csv(file_path: str) -> AddressMap:
    """read addresses from a csv file"""
    address_map = AddressMap()
    reader = csv.reader(open(file_path, 'r'))
    for row in reader:
        address = read_address_row(row)
        address_map.insert(address)
    return address_map


def read_package_row(row: List[str], address_map: AddressMap) -> Package:
    """read a package from a csv row"""
    pack_id = int(row[0])
    address = address_map.get_by_id(int(row[1]))
    deadline = None
    if row[2] != 'EOD':
        deadline = time.fromisoformat(row[2])
    weight = float(row[3])
    status = 'HUB'
    requires_truck = None
    delay = None
    requires_packs = []
    # split special notes into the first word and the rest
    special_notes = row[4].split(' ')
    if special_notes[0] == 'TRUCK':
        requires_truck = int(special_notes[1])
    elif special_notes[0] == 'DELAYED':
        delay = int(special_notes[1])
    elif special_notes[0] == 'PACK':
        requires_packs = [int(pack) for pack in special_notes[1:]]
    return Package(address.address_id, weight, pack_id, deadline, status, requires_truck, None, delay, requires_packs)


def read_package_csv(file_path: str, address_map: AddressMap) -> PackageMap:
    """read packages from a csv file"""
    packages = PackageMap()
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    next(reader)
    for row in reader:
        package = read_package_row(row, address_map)
        packages.insert(package)
    return packages
