from dataclasses import dataclass
from typing import Optional, Union, List
from datetime import time


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
    address = int(row[1])
    deadline = time.fromisoformat(row[2]) if row[2] else None
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
