from address_id import AddressIdTable
from package import PackageTable


address_id_table = AddressIdTable.load_from_file('data/address_id.csv')
package_table = PackageTable.load_from_file('data/package.csv', address_id_table)
