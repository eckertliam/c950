from dataclasses import dataclass
from typing import Optional, Union, List, Callable
from datetime import time
import csv
from address_id import AddressIdTable


@dataclass
class Package:
    """Package class"""
    id: int
    address: int
    weight: float
    deadline: Optional[time] = None
    note: Optional[Union['RequireTruck', 'Delay', 'DeliveredWith']] = None
    truck_id: Optional[int] = None
    status: Optional[str] = None


@dataclass
class RequireTruck:
    """class for packages that require a specific truck to deliver"""
    truck_id: int


@dataclass
class Delay:
    """delayed package class"""
    delay_time: time


@dataclass
class DeliveredWith:
    """class for packages that are delivered with other packages"""
    package_ids: List[int]


def parse_pack_from_row(row: List[str], address_id_table: AddressIdTable) -> Package:
    """Parse a package from a row of a csv reader"""
    pack_id = int(row[0])
    address = address_id_table.get_id(row[1])
    if address is None:
        raise Exception(f'Address {row[1]} not found in address to id table.')
    deadline = row[2]
    if deadline == 'EOD':
        deadline = None
    else:
        deadline = time.fromisoformat(deadline)
    weight = float(row[3])
    raw_note = row[4]
    note = None
    if raw_note:
        # split note by space
        note_parts = raw_note.split(' ')
        if note_parts[0] == 'TRUCK':
            note = RequireTruck(int(note_parts[1]))
        elif note_parts[0] == 'DELAYED':
            note = Delay(time.fromisoformat(note_parts[1]))
        elif note_parts[0] == 'PACK':
            note = DeliveredWith([int(x) for x in note_parts[1:]])
    return Package(pack_id, address, weight, deadline, note)


def parse_packs_from_file(file_path: str, address_id_table: AddressIdTable) -> List[Package]:
    """Parse packages from a csv file"""
    packs = []
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    next(reader)
    for row in reader:
        packs.append(parse_pack_from_row(row, address_id_table))
    return packs


class PackageTable:
    """Package table class: query packages by different criteria"""
    def __init__(self, address_id_table: Optional[AddressIdTable] = None, file_path: Optional[str] = None) -> None:
        """need both address_id_table and file_path or neither to initialize the table"""
        if address_id_table and file_path:
            self.table: List[Package] = parse_packs_from_file(file_path, address_id_table)
        else:
            self.table = []

    def get_package(self, package_id: int) -> Optional[Package]:
        """get a package by id"""
        for pack in self.table:
            if pack.id == package_id:
                return pack
        return None

    def get_special_packages(self) -> List[Package]:
        """get all packages with a note"""
        return [pack for pack in self.table if pack.note is not None]

    def get_deadline_packages(self) -> List[Package]:
        """get all packages with a deadline"""
        return [pack for pack in self.table if pack.deadline is not None]

    def get_delayed_packages(self) -> List[Package]:
        """get all delayed packages"""
        return [pack for pack in self.table if pack.note is not None and isinstance(pack.note, Delay)]

    def get_truck_required_packages(self) -> List[Package]:
        """get all packages that require a specific truck"""
        return [pack for pack in self.table if pack.note is not None and isinstance(pack.note, RequireTruck)]

    def get_packages_by_address(self, address_id: int) -> List[Package]:
        """get all packages going to a specific address"""
        return [pack for pack in self.table if pack.address == address_id]

    def query_packages(self, predicate: Callable[[Package], bool]) -> List[Package]:
        """query packages by a predicate function"""
        return [pack for pack in self.table if predicate(pack)]
