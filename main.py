from typing import List, Tuple
import csv


DistanceMatrix = List[List[float]]


def distance_matrix_from_csv(file_path: str) -> DistanceMatrix:
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


AddressIdMap = List[str]


def address_id_from_csv(file_path: str) -> AddressIdMap:
    """Creates an address id map that is O(1) to access by id"""
    address_id_map: List[Tuple[int, str]] = []
    id_max = 0
    reader = csv.reader(open(file_path, 'r'))
    for row in reader:
        addr_id = int(row[1])
        address_id_map.append((addr_id, row[0]))
        if id > id_max:
            id_max = id
    # create a list of size id_max + 1
    address_id_list = ['' for _ in range(id_max + 1)]
    for addr_id, address in address_id_map:
        address_id_list[addr_id] = address
    return address_id_list
