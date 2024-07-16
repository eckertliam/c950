from address_id import AddressIdTable
from package import PackageTable
from distance import DistanceTable

address_id_table = AddressIdTable('data/address_id.csv')
package_table = PackageTable(address_id_table, 'data/packages.csv')
distance_table = DistanceTable('data/distances.csv')
