from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import time
import csv
from address_id import get_id

@dataclass
class Package:
    id: int
    address: int
    weight: float
    deadline: Optional[time] = None
    note: Optional[Union['RequireTruck', 'Delay', 'DeliveredWith']] = None


@dataclass
class RequireTruck:
    truck_id: int


@dataclass
class Delay:
    delay_time: time


@dataclass
class DeliveredWith:
    package_ids: List[int]


def parse_pack_from_row(row) -> Package:
    pack_id = int(row[0])
    address = get_id(row[1])
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


def parse_packs_from_file(file_path: str) -> List[Package]:
    packs = []
    reader = csv.reader(open(file_path, 'r'))
    # skip header
    next(reader)
    for row in reader:
        packs.append(parse_pack_from_row(row))
    return packs


PACKS = parse_packs_from_file('data/package.csv')

# TODO: write functions to query packages by id, address, deadline, and special notes
