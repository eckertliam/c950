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
    delivery_time: Optional[time] = None


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

    def mark_delivered(self, pack_id: int, time_delivered: time) -> None:
        """mark a package as delivered"""
        package = self.get(pack_id)
        if package:
            package.status = Status.DELIVERED
            package.delivery_time = time_delivered
            self.insert(package)

    def mark_address_delivered(self, address_id: int, time_delivered: time) -> None:
        """mark all packages at an address as delivered"""
        for package in self.packages:
            if package and package.address_id == address_id:
                self.mark_delivered(package.pack_id, time_delivered)
                print(f'Package {package.pack_id} has been delivered')

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
    route: list[int] = field(default_factory=list)
    packages: List[int] = field(default_factory=list)
    visited: List[int] = field(default_factory=list)
    departure_time: time = time(8, 0)
    time_elapsed: timedelta = timedelta(hours=0, minutes=0)
    miles: float = 0
    complete_time: Optional[time] = None

    def __post_init__(self):
        # append hub to the visited list
        self.visited.append(0)

    def add_to_route(self, address_id: int, package_map: PackageMap) -> bool:
        """add an address to the truck route returns true if the truck has room for the address"""
        new_packs = package_map.add_address_to_route(address_id)
        if len(self.packages) + len(new_packs) > TRUCK_MAX_PACKAGES:
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

    def visit(self, address_id: int, package_map: PackageMap, distance_matrix: DistanceMatrix) -> None:
        """visit an address and mark all packages as delivered"""
        print(f'Truck {self.truck_id} is visiting address {address_id}')
        self.visited.append(address_id)
        self.route.remove(address_id)
        # set all packages at the address to delivered
        package_map.mark_address_delivered(address_id, self.calc_current_time(distance_matrix))
        # remove all packages for the address from the truck
        self.packages = [pack_id for pack_id in self.packages if package_map.get(pack_id).address_id != address_id]

    def calc_current_time(self, distance_matrix: DistanceMatrix) -> time:
        """calculate the current relative time of the truck based on the distance matrix"""
        minutes_elapsed = 0
        for i in range(len(self.visited) - 1):
            distance = distance_matrix[self.visited[i]][self.visited[i + 1]]
            minutes_elapsed += distance / TRUCK_SPEED * 60
        # use integer division to get hours
        hours = minutes_elapsed // 60
        # mod the remaining minutes
        minutes = minutes_elapsed % 60
        et = timedelta(hours=hours, minutes=minutes)
        return (datetime.combine(datetime.min, self.departure_time) + et).time()

    def calc_miles_traveled(self, distance_matrix: DistanceMatrix) -> float:
        """calculate the miles traveled by the truck"""
        miles = 0
        for i in range(len(self.visited) - 1):
            miles += distance_matrix[self.visited[i]][self.visited[i + 1]]
        return miles




class TruckMap:
    def __init__(self, size: int = 3):
        self.size = size
        self.trucks: List[Optional[Truck]] = [None] * size
        self._default()

    def _default(self):
        truck1 = Truck(1)
        truck2 = Truck(2)
        truck3 = Truck(3)
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

# CORE FUNCTIONS


def load_trucks(truck_map: TruckMap, package_map: PackageMap, address_map: AddressMap):
    """load trucks with packages based on package requirements and deadlines"""
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


def nearest_address(route: List[int], visited: List[int], distance_matrix: DistanceMatrix) -> Optional[int]:
    """get the nearest address to the last address in the route that has not been visited"""
    closest = 1000
    candidate = -1
    for address_id, distance in enumerate(distance_matrix[visited[-1]]):
        if address_id in route and address_id not in visited and distance < closest:
            closest = distance
            candidate = address_id
    if candidate == -1:
        return None
    return candidate


def next_address(truck: Truck, distance_matrix: DistanceMatrix, package_map: PackageMap) -> Optional[int]:
    """get the next address for the truck using the greedy algorithm but with considerations for deadlines"""
    # check if any packages have a deadline
    deadline_addresses = []
    print(truck.packages)
    for pack_id in truck.packages:
        package = package_map.get(pack_id)
        # make sure the package is not already delivered
        if package.status != Status.DELIVERED and package.deadline and package.address_id not in deadline_addresses:
            deadline_addresses.append(package.address_id)
    if deadline_addresses:
        # get the nearest address with a deadline
        return nearest_address(deadline_addresses, truck.visited, distance_matrix)
    else:
        # get the nearest address without considering deadlines
        return nearest_address(truck.route, truck.visited, distance_matrix)


def step_truck(truck: Truck, distance_matrix: DistanceMatrix, package_map: PackageMap) -> bool:
    """step the truck to the next address"""
    address = next_address(truck, distance_matrix, package_map)
    if address is not None:
        truck.visit(address, package_map, distance_matrix)
        return True
    else:
        print(f'Truck {truck.truck_id} has completed its route')
        return False

# QUERY FUNCTIONS


def _help():
    """Print the help message"""
    print('Commands:')
    print('exit: exit the program')
    print('help: display this help message')
    print('PACKAGE <package_id> | * : get information about a package, or all packages')
    print('TRUCK <truck_id> | * : get information about a truck, or all trucks')
    print('ADDRESS <address_id> | * : get information about an address, or all addresses')
    print('STATUS <status> | * : get all packages with a certain status, or all packages')
    print('<query> AT <time> : queries display the end state of the system, but queries can be done for a time')
    print('Examples:')
    print('PACKAGE 1')
    print('TRUCK 1')
    print('ADDRESS 1')
    print('STATUS DELIVERED')
    print('STATUS HUB')
    print('PACKAGE * AT 10:00')


def query_package(package_map: PackageMap, package_id: int):
    """Query a package by package id"""
    package = package_map.get(package_id)
    if package:
        # TODO: print package info
    else:
        print(f'Package {package_id} not found')


def query_truck(truck_map: TruckMap, truck_id: int):
    """Query a truck by truck id"""
    truck = truck_map.get(truck_id)
    if truck:
        # TODO: print truck info
    else:
        print(f'Truck {truck_id} not found')


def query_address(address_map: AddressMap, address_id: int):
    """Query an address by address id"""
    address = address_map.get_by_id(address_id)
    if address:
        # TODO: print address info
    else:
        print(f'Address {address_id} not found')


def repl(truck_map: TruckMap, address_map: AddressMap, package_map: PackageMap, distance_matrix: DistanceMatrix):
    # a read-eval-print loop for querying the package delivery system
    # syntax inspired by SQL
    # first flush the terminal
    print('\033c')
    print('Welcome to the Package Delivery System Query Interface')
    print('Type "help" for a list of commands')
    print('Type "exit" to quit')
    # wait for user input
    while True:
        query = input('>>> ')
        # split the query by spaces
        query_chunks = query.split(' ')
        # match on the query
        match query_chunks[0]:
            case 'exit':
                break
            case 'help':
                _help()
            case 'PACKAGE':
                if query_chunks[1] == '*':
                    # query all packages
                    pass
                else:
                    query_package(package_map, int(query_chunks[1]))
            case 'TRUCK':
                if query_chunks[1] == '*':
                    # query all trucks
                    pass
                else:
                    query_truck(truck_map, int(query_chunks[1]))
            case 'ADDRESS':
                if query_chunks[1] == '*':
                    # query all addresses
                    pass
                else:
                    query_address(address_map, int(query_chunks[1]))
            case 'STATUS':
                # query packages by status
                pass




def main():
    truck_map = TruckMap()
    address_map = read_address_csv('data/address_id.csv')
    distance_matrix = distance_matrix_from_csv('data/distances.csv')
    package_map = read_package_csv('data/packages.csv', address_map)
    load_trucks(truck_map, package_map, address_map)
    for truck in truck_map.trucks:
        if truck:
            while step_truck(truck, distance_matrix, package_map):
                pass


if __name__ == '__main__':
    main()
