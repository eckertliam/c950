from typing import Optional
import csv


class DistanceTable:
    def __init__(self, file_path: Optional[str]) -> None:
        self.table = []
        if file_path:
            self.load_from_file(file_path)

    def load_from_file(self, file_path: str) -> None:
        reader = csv.reader(open(file_path, 'r'))
        # skip header
        next(reader)
        for row in reader:
            # trim the first element
            row = row[1:]
            new_row = [float(distance) for distance in row]
            self.table.append(new_row)

    def get_distance(self, address_id1: int, address_id2: int) -> Optional[float]:
        try:
            return self.table[address_id1][address_id2]
        except IndexError:
            return None
