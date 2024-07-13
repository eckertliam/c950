from package import Package
from address_id import AddressIdTable
import csv
from datetime import time


def read_packages(addr_id_table: AddressIdTable) -> list[Package]:
    """Reads data/packages.csv and returns a list of Package objects"""
    packages = []
    with open("data/packages.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            pack_id = int(row[0])
            address = addr_id_table[row[1]]
            deadline = None if row[2] == "" else time.fromisoformat(row[2])
            weight = float(row[3])
            packages.append(Package(pack_id, address, weight, deadline))
    return packages
