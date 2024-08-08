import math
from typing import List, Optional, Tuple, TypeVar, Generic
from enum import Enum, auto
import csv
from dataclasses import dataclass, field
from datetime import time, timedelta, datetime

# CONSTANTS

# speed of the truck in miles per hour
TRUCK_SPEED = 18

# truck max packages
TRUCK_MAX_PACKAGES = 16

# truck max miles
TRUCK_MAX_MILES = 140

# truck starting address
TRUCK_HUB = 0

# DATA STRUCTURES

# first we need generic type variables for the map
# K is the key type
K = TypeVar('K')
# V is the value type
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
        # get the bucket index
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
        """
        List all key-value pairs in the map O(N)
        """
        return [item for bucket in self.buckets for item in bucket]

    def __iter__(self):
        return iter(self.list_all())

    def list_keys(self) -> List[K]:
        """List all keys in the map O(N)"""
        return [k for bucket in self.buckets for k, _ in bucket]

    def list_values(self) -> List[V]:
        """List all values in the map O(N)"""
        return [v for bucket in self.buckets for _, v in bucket]

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
    """
    AddressMap is a map of addresses that can be accessed by address string or address id
    """
    def __init__(self):
        self.addresses = Map[str, Address]()
        self.ids = Map[int, Address]()

    def insert(self, address: Address) -> None:
        """Insert an address into the map worst case O(N) best case O(1)"""
        self.addresses.insert(address.address, address)
        self.ids.insert(address.address_id, address)

    def get_by_address(self, address: str) -> Address:
        """Get an address by address string worst case O(N) best case O(1)"""
        return self.addresses.get(address)

    def get_by_id(self, address_id: int) -> Address:
        """Get an address by address id worst case O(N) best case O(1)"""
        return self.ids.get(address_id)


class PackageStatus(Enum):
    """Enum for package status"""
    HUB = auto()
    EN_ROUTE = auto()
    DELIVERED = auto()

    def __repr__(self):
        match self:
            case PackageStatus.HUB:
                return "Hub"
            case PackageStatus.EN_ROUTE:
                return "En Route"
            case PackageStatus.DELIVERED:
                return "Delivered"


@dataclass
class Package:
    """
    Package Dataclass:
    address_id: points to an address in the address map
    weight: the weight of the package
    pack_id: the id of the package in the package map
    deadline: optional deadline for the package
    status: a status enum for the package
    requires_truck: optional truck id that the package requires
    truck_id: a truck id once the package is assigned to a truck
    delay: for packages that are delayed
    depends_on: a list of package ids that this package requires
    is_depended_on: a list of package ids that require this package
    delivery_time: the time the package was delivered
    """
    address_id: int
    weight: float
    pack_id: int
    deadline: Optional[time] = None
    status: PackageStatus = PackageStatus.HUB
    requires_truck: Optional[int] = None
    truck_id: Optional[int] = None
    delay: Optional[time] = None
    depends_on: List[int] = field(default_factory=list)
    delivery_time: Optional[time] = None

    def get_window(self) -> timedelta:
        departure = time(8, 0)
        if self.delay:
            departure = self.delay
        # we cant subtract times from each other so we have to combine them with a date
        departure_dt = datetime.combine(datetime.min, departure)
        deadline = time(17, 0)
        if self.deadline:
            deadline = self.deadline
        deadline_dt = datetime.combine(datetime.min, deadline)
        return deadline_dt - departure_dt

    def tw_is_smaller(self, other: 'Package') -> bool:
        """Check if this package has a smaller time window than another package"""
        return self.get_window() < other.get_window()

    def dep_is_gt(self, other: 'Package') -> bool:
        """Check if this package is depended on more than another package"""
        return len(self.depends_on) > len(other.depends_on)

    def requires_truck_is_gt(self, other: 'Package') -> bool:
        """Check if only one package requires a truck"""
        return self.requires_truck and not other.requires_truck

    def priority_is_gt(self, other: 'Package') -> bool:
        """Check if this package is higher priority than another package"""
        if self.requires_truck_is_gt(other):
            return True
        elif self.tw_is_smaller(other):
            return True
        else:
            return False

    def __gt__(self, other):
        return self.priority_is_gt(other)

    def __lt__(self, other):
        return other.priority_is_gt(self)

    def __eq__(self, other):
        return not self.priority_is_gt(other) and not other.priority_is_gt(self)


# allows for O(1) access to packages by package id
# matrix[package 1][package 2] = distance between package 1 and package 2
DistanceMatrix = List[List[float]]


class PackageMap:
    def __init__(self) -> None:
        self.package_map: Map[int, Package] = Map[int, Package]()
        # package ids sorted by priority
        self.by_priority: List[int] = []

    def insert(self, package: Package) -> None:
        self.package_map.insert(package.pack_id, package)

    def get(self, pack_id: int) -> Optional[Package]:
        return self.package_map.get(pack_id)

    def remove(self, pack_id: int) -> None:
        self.package_map.remove(pack_id)

    def list_values(self) -> List[Package]:
        return self.package_map.list_values()

    def list_keys(self) -> List[int]:
        return self.package_map.list_keys()

    def sort_by_priority(self) -> None:
        # pre-sort the packages by priority
        pre_sort = sorted(self.list_values(), reverse=True)
        for package in pre_sort:
            if package.pack_id not in self.by_priority:
                if package.depends_on:
                    dep_list = [package] + [self.get(dep) for dep in package.depends_on]
                    dep_list = sorted(dep_list, reverse=True)
                    self.by_priority.extend([dep.pack_id for dep in dep_list])
                else:
                    self.by_priority.append(package.pack_id)


@dataclass
class Truck:
    """Truck is a vehicle that delivers packages to addresses
    it has a truck id, a list of package ids it is carrying, and a list of addresses it has visited"""
    truck_id: int
    route: list[int] = field(default_factory=list)
    packages: List[int] = field(default_factory=list)
    visited: List[int] = field(default_factory=list)
    departure_time: time = time(8, 0)
    complete_time: Optional[time] = None
    has_driver: bool = True
    route_distance: float = 0
    time_for_route: timedelta = timedelta(hours=0)

    def __post_init__(self):
        self.route.append(TRUCK_HUB)

    def load(self, package: Package, distance_matrix: DistanceMatrix) -> None:
        # add address to route if not already in route
        if package.address_id not in self.route:
            self.route.append(package.address_id)
            distance = distance_matrix[self.route[-2]][self.route[-1]]
            # add distance to route distance
            self.route_distance += distance
            # minutes to the next address
            minutes = distance / TRUCK_SPEED * 60
            # add time to time for route
            self.time_for_route += timedelta(minutes=minutes)
        self.packages.append(package.pack_id)

    def time_when_route_complete(self) -> time:
        """calculate the time the truck will complete its route as it is"""
        return (datetime.combine(datetime.min, self.departure_time) + self.time_for_route).time()

    def truck_full(self) -> bool:
        """check if the truck is full"""
        return len(self.packages) >= TRUCK_MAX_PACKAGES

    def prep_departure(self) -> None:
        # preps the truck for departure
        # up until now time_for_route is the time it takes to get to the last address in the route
        # during the delivery process time_for_route will be used to calculate the delivery time of packages
        self.time_for_route = timedelta(hours=0)



class TruckMap:
    def __init__(self):
        self.trucks: Map[int, Truck] = Map[int, Truck]()
        self._default()

    def _default(self):
        truck1 = Truck(1)
        truck2 = Truck(3, departure_time=time(9, 5))
        truck3 = Truck(2, departure_time=time(11, 00), has_driver=False)
        self.insert(truck1)
        self.insert(truck2)
        self.insert(truck3)

    def insert(self, truck: Truck) -> None:
        self.trucks.insert(truck.truck_id, truck)

    def get(self, truck_id: int) -> Optional[Truck]:
        return self.trucks.get(truck_id)


# INFINITY
class Infinity:
    """A class to represent positive infinity for integers"""
    def __gt__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __eq__(self, other):
        return False


# infinity constant
INF = Infinity()


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
    status = PackageStatus.HUB
    requires_truck = None
    delay = None
    depends_on = []
    # split special notes into the first word and the rest
    special_notes = row[4].split(' ')
    if special_notes[0] == 'TRUCK':
        requires_truck = int(special_notes[1])
    elif special_notes[0] == 'DELAYED':
        delay = time.fromisoformat(special_notes[1])
    elif special_notes[0] == 'PACK':
        depends_on = [int(pack) for pack in special_notes[1:]]
    return Package(address.address_id, weight, pack_id, deadline, status, requires_truck, None, delay, depends_on)


def read_package_csv(file_path: str, address_map: AddressMap) -> PackageMap:
    """read packages from a csv file"""
    packages = PackageMap()
    reader = csv.reader(open(file_path, 'r'))
    # track the packages that depend on other packages
    dependent_packages = Map[int, List[int]]()
    # skip header
    next(reader)
    for row in reader:
        package = read_package_row(row, address_map)
        packages.insert(package)
        if package.depends_on:
            if package.pack_id not in dependent_packages:
                dependent_packages.insert(package.pack_id, [])
            dependent_packages.get(package.pack_id).extend(package.depends_on)
            # set the packages that depend on other packages
            for pack_id in package.depends_on:
                if pack_id not in dependent_packages:
                    dependent_packages.insert(pack_id, [])
                dependent_packages.get(pack_id).append(package.pack_id)
    # update package dependencies once all packages are read
    for pack_id, dependents in dependent_packages:
        for dep_id in dependents:
            if pack_id not in packages.get(dep_id).depends_on:
                packages.get(dep_id).depends_on.append(pack_id)
    packages.sort_by_priority()
    return packages

# CORE FUNCTIONS


def pack_req_truck(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # make sure truck is not full throw an error if it is
    # O(1) package just gets assigned to the truck
    if truck_map.get(package.requires_truck).truck_full():
        raise ValueError(f'Package {package.pack_id} requires truck {package.requires_truck} but truck is full')
    truck_map.get(package.requires_truck).load(package, distance_matrix)
    package_map.get(package.pack_id).truck_id = package.requires_truck
    return package.requires_truck


def pack_delay_deadline(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # find a truck that can deliver the package before the deadline but leaves after the delay
    # O(N) where N is the number of trucks
    viable_truck = None
    for truck in truck_map.trucks.list_values():
        if truck.time_when_route_complete() <= package.deadline and truck.departure_time >= package.delay and not truck.truck_full():
            if not viable_truck:
                viable_truck = truck
            # truck closest to the package address is the most viable
            elif distance_matrix[truck.route[-1]][package.address_id] < distance_matrix[viable_truck.route[-1]][package.address_id]:
                viable_truck = truck
    if viable_truck:
        truck_map.get(viable_truck.truck_id).load(package, distance_matrix)
        package_map.get(package.pack_id).truck_id = viable_truck.truck_id
        return viable_truck.truck_id
    # if no truck can deliver the package before the deadline but leaves after the delay
    raise ValueError(f'No truck can deliver package {package.pack_id} before the deadline that leaves after the delay')


def pack_delay(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # find a truck that can deliver the package after the delay
    # O(N) where N is the number of trucks
    viable_truck = None
    for truck in truck_map.trucks.list_values():
        if truck.departure_time >= package.delay and not truck.truck_full():
            if not viable_truck:
                viable_truck = truck
            # truck closest to the package address is the most viable
            elif distance_matrix[truck.route[-1]][package.address_id] < distance_matrix[viable_truck.route[-1]][package.address_id]:
                viable_truck = truck
    if viable_truck:
        truck_map.get(viable_truck.truck_id).load(package, distance_matrix)
        package_map.get(package.pack_id).truck_id = viable_truck.truck_id
        return viable_truck.truck_id
    # if no truck can deliver the package after the delay
    raise ValueError(f'No truck can deliver package {package.pack_id} after the delay')


def pack_deadline(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # find a truck that can deliver the package before the deadline
    # O(N) where N is the number of trucks
    viable_truck = None
    for truck in truck_map.trucks.list_values():
        if truck.time_when_route_complete() <= package.deadline and not truck.truck_full():
            if not viable_truck:
                viable_truck = truck
            # truck closest to the package address is the most viable
            elif distance_matrix[truck.route[-1]][package.address_id] < distance_matrix[viable_truck.route[-1]][package.address_id]:
                viable_truck = truck
    if viable_truck:
        truck_map.get(viable_truck.truck_id).load(package, distance_matrix)
        package_map.get(package.pack_id).truck_id = viable_truck.truck_id
        return viable_truck.truck_id
    # if no truck can deliver the package before the deadline
    raise ValueError(f'No truck can deliver package {package.pack_id} before the deadline')


def pack_depends_on(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # because of how we sort the packages by priority
    # we know the packages that depend on this package are right after it in priority and this is the most restrictive
    # by restrictive i mean that the package
    truck_id = assign_pack(package, truck_map, package_map, distance_matrix)
    # get the package that depends on this package
    for pack_id in package.depends_on:
        if package_map.get(pack_id).truck_id == truck_id:
            continue
        if package_map.get(pack_id).truck_id:
            raise ValueError(f'Package {pack_id} depends on package {package.pack_id} but package {pack_id} is already assigned to truck {package_map.get(pack_id).truck_id} but needs to be assigned to truck {truck_id}')
        # assign the package that depends on this package
        truck_map.get(truck_id).load(package_map.get(pack_id), distance_matrix)
        package_map.get(pack_id).truck_id = truck_id
    return truck_id


def greedy_assign(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # if package has no special requirements
    # first check if the package address is already in a truck route
    # if it is not choose the truck that is closest to the package address
    # if the truck is full choose the next closest truck
    # O(N) where N is the number of trucks
    # shortest distance is set to infinity
    shortest_distance = INF
    closest_truck = None
    for truck in truck_map.trucks.list_values():
        if not truck.truck_full():
            if package.address_id in truck.route:
                closest_truck = truck
                break
            if distance_matrix[truck.route[-1]][package.address_id] < shortest_distance:
                shortest_distance = distance_matrix[truck.route[-1]][package.address_id]
                closest_truck = truck
    if closest_truck:
        truck_map.get(closest_truck.truck_id).load(package, distance_matrix)
        package_map.get(package.pack_id).truck_id = closest_truck.truck_id
        return closest_truck.truck_id
    else:
        raise ValueError(f'No truck can deliver package {package.pack_id}')


def assign_pack(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    """
    Use a greedy algorithm to assign a package to a truck return the truck id
    best case O(1) if the package is already assigned to a truck
    worst case O(N * M) where N is the number of trucks and M is the number of packages required by the package
    but M is small and constantly 3. so it is effectively O(N)
    """
    if package.deadline and package.delay:
        return pack_delay_deadline(package, truck_map, package_map, distance_matrix)
    elif package.delay:
        return pack_delay(package, truck_map, package_map, distance_matrix)
    elif package.deadline:
        return pack_deadline(package, truck_map, package_map, distance_matrix)
    elif package.requires_truck:
        return pack_req_truck(package, truck_map, package_map, distance_matrix)
    else:
        return greedy_assign(package, truck_map, package_map, distance_matrix)

def assign_packs(package_map: PackageMap, truck_map: TruckMap, distance_matrix: DistanceMatrix) -> None:
    """
    Assign all packages to trucks
    best case O(N) where N is the number of packages
    worst case is O(N^2) where N is the number of packages
    """
    # assign each package to a truck
    for pack_id in package_map.by_priority:
        try:
            pack = package_map.get(pack_id)
            if pack.truck_id:
                continue
            elif pack.depends_on:
                pack_depends_on(pack, truck_map, package_map, distance_matrix)
            else:
                assign_pack(pack, truck_map, package_map, distance_matrix)
        except ValueError as e:
            # gives a nice error message
            print(e)
            # shows the current state of the trucks in a way to make it easier to debug
    for truck in truck_map.trucks.list_values():
        print(f'Truck {truck.truck_id} Packages: {sorted(truck.packages)} Distance: {truck.route_distance} Time: {truck.time_when_route_complete()}')


def main():
    distance_matrix = distance_matrix_from_csv('data/distances.csv')
    address_map = read_address_csv('data/address_id.csv')
    package_map = read_package_csv('data/packages.csv', address_map)
    truck_map = TruckMap()
    assign_packs(package_map, truck_map, distance_matrix)


if __name__ == '__main__':
    main()
