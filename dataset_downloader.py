import sys
import os
import urllib.request
import tarfile

if not os.path.exists("./datasets"):
    os.mkdir("./datasets")

if not os.path.isdir("./datasets"):
    print("error")
    sys.exit()

os.chdir("./datasets")
url = "https://vision.in.tum.de/old/data/beethoven_data.tar.gz"
file_path = "./beethoven_data.tar.gz"
print("downloading dataset at {path}...".format(path=file_path))
urllib.request.urlretrieve(url, file_path)
print("finished downloading")

print("extracting...")
tar = tarfile.open(file_path, "r:gz")
tar.extractall()
tar.close()
print("success")



