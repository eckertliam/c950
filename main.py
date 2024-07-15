from address_id import AddressIdTable
from package import PackageTable


address_id_table = AddressIdTable('data/address_id.csv')
package_table = PackageTable(address_id_table, 'data/packages.csv')
