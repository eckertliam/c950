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

# allows for O(1) access to packages by package id
# matrix[package 1][package 2] = distance between package 1 and package 2
DistanceMatrix = List[List[float]]

# first we need generic type variables for the map
# K is the key type
K = TypeVar('K')
# V is the value type
V = TypeVar('V')


class Map(Generic[K, V]):
    """Map Class: a simple hash map implementation"""
    def __init__(self, size: int = 40) -> None:
        self.size = size
        self.buckets: List[List[Tuple[K, V]]] = [[] for _ in range(size)]

    def _hash(self, key: K) -> int:
        # private hash function
        return hash(key) % self.size

    def insert(self, key: K, value: V) -> None:
        # insert a key-value pair into the map
        # best case O(1) if bucket is empty
        # worst case O(N) if key already exists where N is the number of key-value pairs in the bucket
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
        # remove a key-value pair from the map
        # best case O(1) if key is at the beginning of the bucket
        # worst case O(N) if key is at the end of the bucket
        idx = self._hash(key)
        for i, (k, _) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx].pop(i)
                return

    def get(self, key: K) -> Optional[V]:
        # get a value by key from the map
        # best case O(1) if key is at the beginning of the bucket
        # worst case O(N) if key is at the end of the bucket
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
        # list all key-value pairs in the map O(N)
        return [item for bucket in self.buckets for item in bucket]

    def __iter__(self):
        return iter(self.list_all())

    def list_keys(self) -> List[K]:
        # list all keys in the map O(N)
        return [k for bucket in self.buckets for k, _ in bucket]

    def list_values(self) -> List[V]:
        # list all values in the map O(N)
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
    # AddressMap is a map of addresses that can be accessed by address string or address id
    def __init__(self):
        self.addresses = Map[str, Address]()
        self.ids = Map[int, Address]()

    def insert(self, address: Address) -> None:
        # Insert an address into the map worst case O(N) best case O(1)
        self.addresses.insert(address.address, address)
        self.ids.insert(address.address_id, address)

    def get_by_address(self, address: str) -> Address:
        # Get an address by address string worst case O(N) best case O(1)
        return self.addresses.get(address)

    def get_by_id(self, address_id: int) -> Address:
        # Get an address by address id worst case O(N) best case O(1)
        return self.ids.get(address_id)


class PackageStatus(Enum):
    # simple enum for package status
    HUB = auto()
    EN_ROUTE = auto()
    DELIVERED = auto()


@dataclass
class Package:
    # Package dataclass
    # address_id points to the address id in the address map
    address_id: int
    weight: float
    pack_id: int
    # optional deadline
    deadline: Optional[time] = None
    # status enum
    status: PackageStatus = PackageStatus.HUB
    # optional truck requirement
    requires_truck: Optional[int] = None
    # truck id once assigned
    truck_id: Optional[int] = None
    # optional delay
    delay: Optional[time] = None
    # packages that depend on this package or that this package depends on
    # built during package reading
    depends_on: List[int] = field(default_factory=list)
    # delivery time
    delivery_time: Optional[time] = None

    def get_window(self) -> timedelta:
        # get an approximation of the time window in which the package can be delivered
        # since packages have not been assigned to trucks yet we have to use the default departure time
        # O(1) runtime complexity
        departure = time(8, 0)
        # delay becomes the departure time if it exists
        if self.delay:
            departure = self.delay
        # we cant subtract times from each other so we have to combine them with a date
        departure_dt = datetime.combine(datetime.min, departure)
        deadline = time(17, 0)
        if self.deadline:
            deadline = self.deadline
        deadline_dt = datetime.combine(datetime.min, deadline)
        # return the time window as a timedelta
        return deadline_dt - departure_dt


class PackageMap:
    def __init__(self) -> None:
        # package map is a map of packages that can be accessed by package id
        self.package_map: Map[int, Package] = Map[int, Package]()
        # packages sorted by priority
        self.by_priority: List[int] = []

    def insert(self, package: Package) -> None:
        # insert a package into the map O(N) worst case O(1) best case
        self.package_map.insert(package.pack_id, package)

    def get(self, pack_id: int) -> Optional[Package]:
        # get a package by package id O(N) worst case O(1) best case
        return self.package_map.get(pack_id)

    def remove(self, pack_id: int) -> None:
        # remove a package by package id O(N) worst case O(1) best case
        self.package_map.remove(pack_id)

    def list_values(self) -> List[Package]:
        # list all values in the map O(N)
        return self.package_map.list_values()

    def list_keys(self) -> List[int]:
        # list all keys in the map O(N)
        return self.package_map.list_keys()

    def sort_by_priority(self, distance_matrix: DistanceMatrix) -> None:
        # sort the packages by priority O(N log N) where N is the number of packages
        sorted_packs = sorted(self.package_map.list_values(),
                              key=lambda pack: (pack.requires_truck is None,
                                                pack.get_window(),
                                                distance_matrix[TRUCK_HUB][pack.address_id]))
        # flatten sorted_packs to a list of package ids
        self.by_priority = [pack.pack_id for pack in sorted_packs]


@dataclass
class Truck:
    # Truck dataclass
    truck_id: int
    # route is a list of address ids to be visited
    route: list[int] = field(default_factory=list)
    # packages is a list of package ids to be delivered
    packages: List[int] = field(default_factory=list)
    # visited is a list of address ids that have been visited
    visited: List[int] = field(default_factory=list)
    departure_time: time = time(8, 0)
    has_driver: bool = True
    route_distance: float = 0
    time_for_route: timedelta = timedelta(hours=0)

    def __post_init__(self):
        # after truck is initialized append the truck hub to the route
        # this helps later in the algo to determine distance to the first address
        # later we pop it off the route before actually stepping through the route
        self.route.append(TRUCK_HUB)

    def load(self, package: Package, distance_matrix: DistanceMatrix) -> None:
        # add address to route if not already in route O(1) runtime complexity
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
        # get the time when the route is complete O(1)
        # time objects can't be used in arithmetic so we have to combine them with a datetime and then extract the time
        return (datetime.combine(datetime.min, self.departure_time) + self.time_for_route).time()

    def truck_full(self) -> bool:
        # check if the truck is full O(1)
        return len(self.packages) >= TRUCK_MAX_PACKAGES

    def step_through_route(self, package_map: PackageMap, distance_matrix: DistanceMatrix) -> None:
        # reset time for route we'll use it to set package delivery times
        # O(M * N) where M is the number of addresses and N is the number of packages
        self.time_for_route = timedelta(hours=0)
        # first move hub to visited since we are at the hub
        self.visited.append(TRUCK_HUB)
        # pop it off the route
        self.route.pop(0)
        for address in self.route:
            # add the address to the visited list
            self.visited.append(address)
            # get the distance travelled to the address
            distance = distance_matrix[self.visited[-2]][address]
            # get the time travelled to the address
            minutes = distance / TRUCK_SPEED * 60
            # add the time to the time for route
            self.time_for_route += timedelta(minutes=minutes)
            pack_delivery_time = self.time_when_route_complete()
            for pack_id in self.packages:
                pack = package_map.get(pack_id)
                if pack.address_id == address:
                    # set the delivery time
                    package_map.get(pack_id).delivery_time = pack_delivery_time



class TruckMap:
    # TruckMap is a a hash map of trucks that can be accessed by truck id
    # it acts as a wrapper around a simple hash map
    def __init__(self):
        self.trucks: Map[int, Truck] = Map[int, Truck]()
        self._default()

    def _default(self):
        truck1 = Truck(1)
        # sets off after truck1 is finished since it has no driver
        truck2 = Truck(2, departure_time=time(11, 00), has_driver=False)
        # sets off when most delayed packs arrive
        truck3 = Truck(3, departure_time=time(9, 5))
        self._insert(truck1)
        self._insert(truck2)
        self._insert(truck3)

    def _insert(self, truck: Truck) -> None:
        # insert is a protected function because we have a constant number of trucks O(N) worst case O(1) best case
        self.trucks.insert(truck.truck_id, truck)

    def get(self, truck_id: int) -> Optional[Truck]:
        # get a truck by truck id O(N) worst case O(1) best case
        return self.trucks.get(truck_id)

    def step_through_routes(self, package_map: PackageMap, distance_matrix: DistanceMatrix) -> None:
        # step through all the routes O(T * (M * N)) where T is the number of trucks, M is the number of addresses and N is the number of packages
        for truck in self.trucks.list_values():
            truck.step_through_route(package_map, distance_matrix)


# INFINITY
class Infinity:
    # infinity class for use in comparisons
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
    # read a distance matrix from a csv file
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
    # read an address from a csv row
    return Address(row[0], row[2], row[1], int(row[3]))


def read_address_csv(file_path: str) -> AddressMap:
    # read addresses from a csv file
    address_map = AddressMap()
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    for row in reader:
        address = read_address_row(row)
        # insert the address into the address map
        address_map.insert(address)
    return address_map


def read_package_row(row: List[str], address_map: AddressMap) -> Package:
    # read a package from a csv row
    pack_id = int(row[0])
    # fetch the address from the address map
    address = address_map.get_by_address(row[1])
    # parse the deadline if it exists
    deadline = None
    if row[2] != 'EOD':
        deadline = time.fromisoformat(row[2])
    # get weight (Note: weights are all ints in the csv but float seems like good practice)
    weight = float(row[3])
    # all packages start at the hub
    status = PackageStatus.HUB
    # special notes
    requires_truck = None
    delay = None
    depends_on = []
    # split special notes into the first word and the rest
    special_notes = row[4].split(' ')
    if special_notes[0] == 'TRUCK':
        # if the package requires a truck parse the truck id
        requires_truck = int(special_notes[1])
    elif special_notes[0] == 'DELAYED':
        # if the package is delayed use time to read the time from the string
        delay = time.fromisoformat(special_notes[1])
    elif special_notes[0] == 'PACK':
        # packages depend on multiple packages so we chop the prefix off and parse the rest using a list comprehension
        depends_on = [int(pack) for pack in special_notes[1:]]
    return Package(address.address_id, weight, pack_id, deadline, status, requires_truck, None, delay, depends_on)


def read_package_csv(file_path: str, address_map: AddressMap, distance_matrix: DistanceMatrix) -> PackageMap:
    # read packages from a csv file
    packages = PackageMap()
    reader = csv.reader(open(file_path, 'r'))
    # track the packages that depend on other packages
    dependent_packages = Map[int, List[int]]()
    # skip header
    next(reader)
    for row in reader:
        # get package and push it into the package map
        package = read_package_row(row, address_map)
        packages.insert(package)
        # if the package depends on other packages
        if package.depends_on:
            # set the packages that this package depends on
            if package.pack_id not in dependent_packages:
                dependent_packages.insert(package.pack_id, [])
            dependent_packages.get(package.pack_id).extend(package.depends_on)
            # add this package_id to the packages that depend on the packages this package depends on
            # this builds a dependency graph that helps when we sort the packages by priority
            for pack_id in package.depends_on:
                if pack_id not in dependent_packages:
                    dependent_packages.insert(pack_id, [])
                dependent_packages.get(pack_id).append(package.pack_id)
    # update package dependencies within the package map once all packages are read
    for pack_id, dependents in dependent_packages:
        for dep_id in dependents:
            if pack_id not in packages.get(dep_id).depends_on:
                packages.get(dep_id).depends_on.append(pack_id)
    packages.sort_by_priority(distance_matrix)
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
    # by restrictive i mean the package which has the smallest time window and the most dependencies
    # O(M) where M is the number of packages that depend on this package
    truck_id = assign_pack(package, truck_map, package_map, distance_matrix)
    # get the packages that depend on this package
    for pack_id in package.depends_on:
        # if the package is already assigned to the truck that is delivering this package then we can skip it
        if package_map.get(pack_id).truck_id == truck_id:
            continue
        # if the package that depends on this package is already assigned to a truck then we have a problem
        # because of how we sort the packages by priority this should never happen but we check just in case
        if package_map.get(pack_id).truck_id:
            raise ValueError(f'Package {pack_id} depends on package {package.pack_id} but package {pack_id} is already assigned to truck {package_map.get(pack_id).truck_id} but needs to be assigned to truck {truck_id}')
        # assign the package that depends on this package
        truck_map.get(truck_id).load(package_map.get(pack_id), distance_matrix)
        package_map.get(pack_id).truck_id = truck_id
    return truck_id


def nearest_neighbor(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # if it is not choose the truck that is closest to the package address
    # if the truck is full choose the next closest truck
    # O(N) where N is the number of trucks
    # shortest distance is set to infinity
    shortest_distance = INF
    closest_truck = None
    for truck in truck_map.trucks.list_values():
        if not truck.truck_full():
            # if package address is in the truck route then the package can be added with no extra distance
            if package.address_id in truck.route:
                closest_truck = truck
                break
            # otherwise if the truck is closer to the package address it is the most viable
            distance = distance_matrix[truck.route[-1]][package.address_id]
            if distance < shortest_distance:
                shortest_distance = distance
                closest_truck = truck
    if closest_truck:
        truck_map.get(closest_truck.truck_id).load(package, distance_matrix)
        package_map.get(package.pack_id).truck_id = closest_truck.truck_id
        return closest_truck.truck_id
    else:
        raise ValueError(f'No truck can deliver package {package.pack_id}')


def assign_pack(package: Package, truck_map: TruckMap, package_map: PackageMap, distance_matrix: DistanceMatrix) -> int:
    # O(N) where N is the number of trucks
    # just dispatch the package to the handler that is most appropriate
    # if it has no special requirements use the nearest neighbor algorithm
    if package.deadline and package.delay:
        return pack_delay_deadline(package, truck_map, package_map, distance_matrix)
    elif package.delay:
        return pack_delay(package, truck_map, package_map, distance_matrix)
    elif package.deadline:
        return pack_deadline(package, truck_map, package_map, distance_matrix)
    elif package.requires_truck:
        return pack_req_truck(package, truck_map, package_map, distance_matrix)
    else:
        return nearest_neighbor(package, truck_map, package_map, distance_matrix)


def assign_packs(package_map: PackageMap, truck_map: TruckMap, distance_matrix: DistanceMatrix) -> None:
    # worst case O(P * M) where P is the total number packages and M is the number of packages that depend on those packages
    # best case O(P * N) where N is the number of trucks which is constant at 3 so O(P)
    errors = []
    # assign each package to a truck
    for pack_id in package_map.by_priority:
        try:
            pack = package_map.get(pack_id)
            # because of how
            if pack.truck_id:
                continue
            elif pack.depends_on:
                # O(M) where M is the number of packages that depend on this package
                pack_depends_on(pack, truck_map, package_map, distance_matrix)
            else:
                # O(N) where N is the number of trucks
                assign_pack(pack, truck_map, package_map, distance_matrix)
        except ValueError as e:
            # lets all errors pass through before stopping the program
            errors.append(e)
    if errors:
        print('Errors occurred while loading packages:')
        for error in errors:
            print(error)
        # shows the current state of the trucks in a way to make it easier to debug
        for truck in truck_map.trucks.list_values():
            print(
                f'Truck {truck.truck_id} Packages: {sorted(truck.packages)} Distance: {truck.route_distance} Time: {truck.time_when_route_complete()}')
        # exit with error
        exit(1)
    # add all truck miles together
    total_miles = 0
    for truck in truck_map.trucks.list_values():
        total_miles += truck.route_distance
    print(f'Total Miles: {total_miles}')


# TUI FUNCTIONS

def get_time() -> time:
    # clear the screen using ANSI escape codes
    print('\033[H\033[J', end='', flush=True)
    # wait for user input
    while True:
        try:
            time_str = input('Enter the time in the format HH:MM enter "exit" to end the program: ')
            if time_str == 'exit':
                exit(0)
            return time.fromisoformat(time_str)
        except ValueError:
            print('Invalid time format')


def dump_package_state(package_map: PackageMap, truck_map: TruckMap) -> None:
    # first we get the time from the user
    user_time = get_time()
    for pack in package_map.list_values():
        truck = truck_map.get(pack.truck_id)
        # if user time is before truck departure time then the package is at the hub
        if user_time < truck.departure_time:
            print(f'Package: {pack.pack_id} At Hub on Truck {truck.truck_id}')
        elif pack.delivery_time and user_time > pack.delivery_time:
            # if user time is after the delivery time then the package was delivered
            print(f'Package {pack.pack_id} was delivered at {pack.delivery_time} by Truck {truck.truck_id}')
        else:
            # otherwise we know the package is en route
            print(f'Package {pack.pack_id} is en route on Truck {truck.truck_id}')
    # recursively call the function
    dump_package_state(package_map, truck_map)


def main():
    distance_matrix = distance_matrix_from_csv('data/distances.csv')
    address_map = read_address_csv('data/address_id.csv')
    package_map = read_package_csv('data/packages.csv', address_map, distance_matrix)
    for pack in package_map.by_priority:
        print(pack)
    truck_map = TruckMap()
    assign_packs(package_map, truck_map, distance_matrix)
    truck_map.step_through_routes(package_map, distance_matrix)
    dump_package_state(package_map, truck_map)


if __name__ == '__main__':
    main()
