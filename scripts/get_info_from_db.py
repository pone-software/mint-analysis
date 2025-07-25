from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
from dbetto import AttrsDict
import sys

def get_data(database='tum', 
             collection='Devices',
             query:dict = {'uid':{"$regex": 'p-1-1-om-hs-*'} },
             ssh_host = 'production.pacific-neutrino.org',
             ssh_user = '',
             ssh_password = '',
             ssh_port = 22,
             remote_bind_address = ('127.0.0.1', 27017),  # MongoDB on remote server
             mongo_username = '',
             mongo_password = '',
             ) -> AttrsDict:

    result = {}
    # Start SSH tunnel
    with SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=remote_bind_address,
        local_bind_address=('127.0.0.1', 27017)  # optional: auto-assign with (None, some_port)
    ) as tunnel:
        
        # Connect to MongoDB via the forwarded local port
        client = MongoClient(
            f'mongodb://{mongo_username}:{mongo_password}@{'127.0.0.1'}:{tunnel.local_bind_port}'
        )

        db = client[database]
        for doc in db[collection].find(query):
            id = str(doc['_id'])
            if id in result.keys():
                raise RuntimeError(f"uid {id} defined multiple times - Check collection {collection} in database {database} sanity.")
            
            del doc['_id']
            if 'subdevices' in doc.keys() and type(doc['subdevices']) == list:
                doc['subdevices'] = {i['uid']:i for i in doc['subdevices']}
                    
            result[id]=doc
        
    return AttrsDict(dict(sorted(result.items())))

if __name__ == "__main__":
    arguments = sys.argv[1:]
    hemisphere = arguments[0] #'p-1-1-om-hs-31'
    db_devices =get_data(query={'uid':{"$regex": 'p-1-1-*'} })
    db_measurements = get_data(collection='Measurements_Pmt', query={'measurement_location':'TUM', 'measurement_type': 'Nominal voltage'})

    pmts = db_devices.map('uid',unique=False)[hemisphere][0].subdevices.map('device_type',unique=False)['pmt-unit'].map('uid')
    settings = AttrsDict()
    for k in pmts.keys():
        val = db_measurements.group('devices_used.uid')[k][0].result
        unc = db_measurements.group('devices_used.uid')[k][0].result_unc
        unit = db_measurements.group('devices_used.uid')[k][0].units

        settings[k] = {"position":int(pmts[k].position),"voltage":val}
    print(settings)