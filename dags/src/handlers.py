import pickle 


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Pickler:
    @staticmethod
    def save(filename, data):
        with open(f"{filename}", "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def read(filename):
        with open(f"{filename}", "rb") as file:
            data = pickle.load(file)
        return data

pickler = Pickler()