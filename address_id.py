import csv
from map import Map
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AddressIdTable:
    table: Map[str, int] = field(default_factory=Map)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'AddressIdTable':
        address_id_table = cls()
        reader = csv.reader(open(file_path, 'r'))
        for row in reader:
            address_id_table.table.push(row[0], int(row[1]))
        return address_id_table

    def get_id(self, address: str) -> Optional[int]:
        return self.table.get(address)

    def get_address(self, address_id: int) -> Optional[str]:
        for key, value in self.table:
            if value == address_id:
                return key
        return None
