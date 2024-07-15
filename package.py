from dataclasses import dataclass
from typing import Optional, Union, List, Callable
from datetime import time
import csv
from address_id import AddressIdTable

@dataclass
class Package:
    id: int
    address: int
    weight: float
    deadline: Optional[time] = None
    note: Optional[Union['RequireTruck', 'Delay', 'DeliveredWith']] = None
    truck_id: Optional[int] = None
    status: Optional[str] = None


@dataclass
class RequireTruck:
    truck_id: int


@dataclass
class Delay:
    delay_time: time


@dataclass
class DeliveredWith:
    package_ids: List[int]


def parse_pack_from_row(row: List[str], address_id_table: AddressIdTable) -> Package:
    pack_id = int(row[0])
    address = address_id_table.get_id(row[1])
    if address is None:
        raise Exception(f'Address {row[1]} not found')
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
    packs = []
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    next(reader)
    for row in reader:
        packs.append(parse_pack_from_row(row, address_id_table))
    return packs


@dataclass
class PackageTable:
    table: List[Package] = None

    @classmethod
    def load_from_file(cls, file_path: str, address_id_table: AddressIdTable) -> 'PackageTable':
        return cls(parse_packs_from_file(file_path, address_id_table))

    def get_package(self, package_id: int) -> Optional[Package]:
        for pack in self.table:
            if pack.id == package_id:
                return pack
        return None

    def get_special_packages(self) -> List[Package]:
        return [pack for pack in self.table if pack.note is not None]

    def get_deadline_packages(self) -> List[Package]:
        return [pack for pack in self.table if pack.deadline is not None]

    def get_delayed_packages(self) -> List[Package]:
        return [pack for pack in self.table if pack.note is not None and isinstance(pack.note, Delay)]

    def get_truck_required_packages(self) -> List[Package]:
        return [pack for pack in self.table if pack.note is not None and isinstance(pack.note, RequireTruck)]

    def get_packages_by_address(self, address_id: int) -> List[Package]:
        return [pack for pack in self.table if pack.address == address_id]

    def query_packages(self, pred: Callable) -> List[Package]:
        return [pack for pack in self.table if pred(pack)]