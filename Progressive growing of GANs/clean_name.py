import os
import shutil

to_remove = ["network-snapshot-"]


for (root,dirs,files) in os.walk('./results'):
    for i in to_remove:
        if i in root:
            shutil.rmtree(root)
            print("remove", root)
