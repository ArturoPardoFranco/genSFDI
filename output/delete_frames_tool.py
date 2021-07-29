'''
Frames and Reconframes deletion tool
Arturo Pardo, 2020
'''

import glob, os, sys

# We will just enter each individual folder in the following directory
path = sys.argv[1]
print('Target path: ' + path)

# And then search inside
folders = glob.glob(path +'/*')
for folder in folders:
    print('Folder: ' + str(folder))
    for file in glob.glob(folder + '/frames/*.png'):
        print('     File: ' + str(file) )
        os.remove(file)
    for file in glob.glob(folder + '/reconframes/*.png'):
        print('     File: ' + str(file))
        os.remove(file)


