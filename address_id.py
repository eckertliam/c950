import csv
from map import Map
from typing import Optional


class AddressIdTable:
    def __init__(self, file_path: Optional[str] = None) -> None:
        self.table = Map[str, int]()
        if file_path:
            self.load_from_file(file_path)

    def load_from_file(self, file_path: str) -> None:
        reader = csv.reader(open(file_path, 'r'))
        for row in reader:
            self.table.set(row[0], int(row[1]))

    def get_id(self, address: str) -> Optional[int]:
        return self.table.get(address)

    def get_address(self, address_id: int) -> Optional[str]:
        for key, value in self.table:
            if value == address_id:
                return key
        return None
