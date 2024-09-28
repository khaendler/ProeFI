import os
import pickle


def save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def load_all_from_folder(folder_path):
    objects = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        obj = load(file_path)
        objects.append(obj)
    return objects

