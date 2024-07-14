import csv
from map import Map
from typing import Optional


class AddressIdTable:
    def __init__(self):
        self.table: Map[str, int] = Map()

    def load_from_file(self, file_path: str) -> None:
        reader = csv.reader(open(file_path, 'r'))
        for row in reader:
            self.table.push(row[0], int(row[1]))

    def get_id(self, address: str) -> Optional[int]:
        return self.table.get(address)

    def get_address(self, address_id: int) -> Optional[str]:
        for key, value in self.table:
            if value == address_id:
                return key
        return None


ADDRESS_ID_TABLE = AddressIdTable()
ADDRESS_ID_TABLE.load_from_file('data/address_id.csv')


def get_id(address: str) -> Optional[int]:
    return ADDRESS_ID_TABLE.get_id(address)


def get_address(address_id: int) -> Optional[str]:
    return ADDRESS_ID_TABLE.get_address(address_id)
