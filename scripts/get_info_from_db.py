import sys

from dbetto import AttrsDict
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder


def get_data(
    query,
    database="tum",
    collection="Devices",
    ssh_host="production.pacific-neutrino.org",
    ssh_user="",
    ssh_password="",
    ssh_port=22,
    remote_bind_address=("127.0.0.1", 27017),  # MongoDB on remote server
    mongo_username="",
    mongo_password="",
) -> AttrsDict:
    """queries production database

    Parameters
    ----------
    database : str
        Database to query
    collection : str
        Collection to query
    query : dict
        MongoDB query
    ssh_host : str
        Address of remote server.
    ssh_port : int
        Port of remote server
    remote_bind_address : tuple
        Tuple of mongoDB binds on local machine (address , port)
    mongo_username : str
        username of the mongoDB
    mongo_password : str
        password of the mongoDB

    Returns
    -------
    AttrsDict
        Attributized dict with entries matching the query result {mongoid:db_entry}
    """
    result = {}
    # Start SSH tunnel
    with SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=remote_bind_address,
        local_bind_address=("127.0.0.1", 27017),  # optional: auto-assign with (None, some_port)
    ) as tunnel:

        # Connect to MongoDB via the forwarded local port
        client = MongoClient(
            f"mongodb://{mongo_username}:{mongo_password}@{'127.0.0.1'}:{tunnel.local_bind_port}"
        )

        db = client[database]
        for doc in db[collection].find(query):
            id = str(doc["_id"])
            if id in result:
                msg = f"uid {id} defined multiple times in {collection} with database {database}"
                raise RuntimeError(msg)

            del doc["_id"]
            if "subdevices" in doc and isinstance(doc["subdevices"], list):
                doc["subdevices"] = {i["uid"]: i for i in doc["subdevices"]}

            result[id] = doc

    return AttrsDict(dict(sorted(result.items())))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    hemisphere = arguments[0]  #'p-1-1-om-hs-31'
    db_devices = get_data(query={"uid": {"$regex": "p-1-1-*"}})
    db_measurements = get_data(
        collection="Measurements_Pmt",
        query={"measurement_location": "TUM", "measurement_type": "Nominal voltage"},
    )

    pmts = (
        db_devices.map("uid")[hemisphere]
        .subdevices.map("device_type", unique=False)["pmt-unit"]
        .map("uid")
    )
    settings = AttrsDict()
    for k in pmts:
        val = db_measurements.group("devices_used.uid")[k][0].result
        unc = db_measurements.group("devices_used.uid")[k][0].result_unc
        unit = db_measurements.group("devices_used.uid")[k][0].units

        settings[k] = {"position": int(pmts[k].position), "voltage": val}
