from typing import List, Optional, Tuple, TypeVar, Generic
from enum import Enum, auto
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
        self.addresses = Map[str, Address]()
        self.ids = Map[int, Address]()

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


class Status(Enum):
    """Enum for package status"""
    HUB = auto()
    EN_ROUTE = auto()
    DELIVERED = auto()


@dataclass
class Package:
    """Package points to an address, package weight in pounds, package id, delivery deadline
    status of the package, a required truck id if given, a truck id if assigned,
    a delay time if the package is delayed, and a list of package ids it must be delivered with if any"""
    address_id: int
    weight: float
    pack_id: int
    deadline: Optional[time] = None
    status: Status = Status.HUB
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
            package.status = Status.DELIVERED
            self.insert(package)

    def mark_en_route(self, pack_id: int) -> None:
        """mark a package as en route"""
        package = self.get(pack_id)
        if package:
            package.status = Status.EN_ROUTE
            self.insert(package)

    def mark_packs_en_route(self, pack_ids: List[int]) -> None:
        """mark a list of packages as en route"""
        for pack_id in pack_ids:
            self.mark_en_route(pack_id)

    def add_address_to_route(self, address_id: int) -> List[int]:
        """get all packages at an address that are not en route and add them to the route"""
        packs = []
        for package in self.packages:
            if package and package.address_id == address_id and package.status == Status.HUB:
                packs.append(package.pack_id)
        return packs

    def get_all_packages(self) -> List[Package]:
        """get all packages in the map"""
        return [package for package in self.packages if package]

    def get_all_hub_packages(self) -> List[Package]:
        """get all packages in the map that are at the hub"""
        return [package for package in self.packages if package and package.status == Status.HUB]


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
    departure_time: time = time(8, 0)

    def add_to_route(self, address_id: int, package_map: PackageMap) -> bool:
        """add an address to the truck route returns true if the truck has room for the address"""
        new_packs = package_map.add_address_to_route(address_id)
        if len(self.packages) + len(new_packs) > self.max_packs:
            return False
        self.packages += new_packs
        self.route.append(address_id)
        package_map.mark_packs_en_route(new_packs)
        return True

    def set_departure_time(self, dt: time, package_map: PackageMap) -> bool:
        """set departure time and check for conflicting delays"""
        self.departure_time = dt
        # loop through packages and make sure they none have conflicting delays
        for pack_id in self.packages:
            package = package_map.get(pack_id)
            if package.delay:
                if dt < package.delay:
                    return False
        return True


class TruckMap:
    def __init__(self, size: int = 3):
        self.size = size
        self.trucks: List[Optional[Truck]] = [None] * size
        self._default()

    def _default(self):
        truck1 = Truck(1, True)
        truck2 = Truck(2, True)
        truck3 = Truck(3, False)
        self.insert(truck1)
        self.insert(truck2)
        self.insert(truck3)

    def _expand(self, new_size: int) -> None:
        self.trucks += [None] * (new_size - self.size)
        self.size = new_size

    def insert(self, truck: Truck) -> None:
        if self.size <= truck.truck_id:
            self._expand(truck.truck_id + 1)
        self.trucks[truck.truck_id] = truck

    def from_list(self, trucks: List[Truck]) -> None:
        for truck in trucks:
            self.insert(truck)

    def get(self, truck_id: int) -> Optional[Truck]:
        return self.trucks[truck_id]

    def add_to_truck_route(self, truck_id: int, address_id: int, package_map: PackageMap) -> bool:
        truck = self.get(truck_id)
        if truck:
            return truck.add_to_route(address_id, package_map)
        return False


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
    """read addresses from a csv file and input the data into an address map"""
    address_map = AddressMap()
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    for row in reader:
        address = read_address_row(row)
        address_map.insert(address)
    return address_map


def read_package_row(row: List[str], address_map: AddressMap) -> Package:
    """read a package from a csv row"""
    pack_id = int(row[0])
    address = address_map.get_by_address(row[1])
    deadline = None
    if row[2] != 'EOD':
        deadline = time.fromisoformat(row[2])
    weight = float(row[3])
    status = Status.HUB
    requires_truck = None
    delay = None
    requires_packs = []
    # split special notes into the first word and the rest
    special_notes = row[4].split(' ')
    if special_notes[0] == 'TRUCK':
        requires_truck = int(special_notes[1])
    elif special_notes[0] == 'DELAYED':
        delay = time.fromisoformat(special_notes[1])
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


def load_trucks(truck_map: TruckMap, package_map: PackageMap, address_map: AddressMap):
    # tracks the addresses that have not been assigned
    addresses = address_map.ids.list_keys()
    # list all packages that require other packages
    # each element is either none or a list of package ids that the package requires
    requires_packs: List[Optional[List[int]]] = [None] * len(package_map.packages)
    # list all packages that have a deadline at their index
    deadlines: List[Optional[time]] = [None] * len(package_map.packages)
    # list all packages that have a delay at their index
    delays: List[Optional[time]] = [None] * len(package_map.packages)
    for package in package_map.get_all_packages():
        if package.requires_truck:
            # if the package requires a truck add it to the truck list
            truck_map.add_to_truck_route(package.requires_truck, package.address_id, package_map)
            addresses.remove(package.address_id)
        if package.requires_packs:
            requires_packs[package.pack_id] = package.requires_packs
        if package.deadline:
            deadlines[package.pack_id] = package.deadline
        if package.delay:
            delays[package.pack_id] = package.delay
    # truck 1 will leave at 0800
    truck_map.get(1).set_departure_time(time(8, 0), package_map)
    # truck 2 will leave at 0930
    truck_map.get(2).set_departure_time(time(9, 30), package_map)
    # truck 3 will leave at 1030
    truck_map.get(3).set_departure_time(time(10, 30), package_map)
    for i, deadline in enumerate(deadlines):
        if deadline and package_map.get(i) and package_map.get(i).address_id in addresses:
            # assign packages that have a deadline and no delay the truck with the earliest departure time
            if not delays[i]:
                if truck_map.add_to_truck_route(1, package_map.get(i).address_id, package_map):
                    addresses.remove(package_map.get(i).address_id)
                elif truck_map.add_to_truck_route(2, package_map.get(i).address_id, package_map):
                    addresses.remove(package_map.get(i).address_id)
                else:
                    raise ValueError(f'Package {i} could not be assigned to a truck that leaves before its deadline')
            else:
                # assign packages that have a delay and deadline to the truck that has a departure time after the delay
                if truck_map.add_to_truck_route(2, package_map.get(i).address_id, package_map):
                    addresses.remove(package_map.get(i).address_id)
                else:
                    raise ValueError(f'Package {i} could not be assigned to a truck that leaves after its delay')

    # assign all remaining packages where they fit and work around delays and packages requiring other packages
    for package in package_map.get_all_hub_packages():
        if package.address_id not in addresses:
            continue
        if not requires_packs[package.pack_id] and not delays[package.pack_id]:
            # see where the package fits
            for truck_id in [1, 2, 3]:
                if truck_map.add_to_truck_route(truck_id, package.address_id, package_map):
                    addresses.remove(package.address_id)
                    break
            # if the package was not assigned to a truck raise an error
            if package.status == Status.HUB:
                raise ValueError(f'Package {package.pack_id} could not be assigned to a truck')
        elif requires_packs[package.pack_id]:
            for dep_pack_id in requires_packs[package.pack_id]:
                dep_pack = package_map.get(dep_pack_id)
                if dep_pack.status != Status.HUB:
                    raise ValueError(f'Package {package.pack_id} requires package {dep_pack_id} which is not at the hub')
                else:
                    # add the packages to truck 3 as it should be the most empty
                    # throws an error if the package cannot be added to a truck
                    if not truck_map.add_to_truck_route(3, package.address_id, package_map):
                        raise ValueError(f'Package {package.pack_id} could not be assigned to truck 3')
                    else:
                        addresses.remove(package.address_id)
            # add the package to truck 3
            # throws an error if the package cannot be added to a truck
            if not truck_map.add_to_truck_route(3, package.address_id, package_map):
                raise ValueError(f'Package {package.pack_id} could not be assigned to truck 3')
            else:
                addresses.remove(package.address_id)
        elif delays[package.pack_id]:
            # add the package to truck 2
            # throws an error if the package cannot be added to a truck
            if not truck_map.add_to_truck_route(2, package.address_id, package_map):
                if not truck_map.add_to_truck_route(3, package.address_id, package_map):
                    raise ValueError(f'Package {package.pack_id} could not be assigned to truck 2 or 3')
            addresses.remove(package.address_id)


def main():
    truck_map = TruckMap()
    address_map = read_address_csv('data/address_id.csv')
    distance_matrix = distance_matrix_from_csv('data/distances.csv')
    package_map = read_package_csv('data/packages.csv', address_map)
    load_trucks(truck_map, package_map, address_map)
    print('Truck 1')
    print(truck_map.get(1).route)
    print('Truck 2')
    print(truck_map.get(2).route)
    print('Truck 3')
    print(truck_map.get(3).route)
    # check if any packages are not en route
    for package in package_map.get_all_packages():
        if package.status != Status.EN_ROUTE:
            print(f'Package {package.pack_id} is not en route')


if __name__ == '__main__':
    main()
