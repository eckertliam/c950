import csv


class AddressIdTable:
    """
    Reads in the file address_id.csv and provides a mapping between address and id, where index in the list is the id and
    the value is the address.
    """
    def __init__(self):
        self.table: list[str] = []
        with open("data/address_id.csv") as file:
            reader = csv.reader(file)
            for row in reader:
                self.table.append(row[0])

    def __index__(self, index: int) -> str:
        return self.table[index]

    def __getitem__(self, address: str) -> int:
        return self.table.index(address)
