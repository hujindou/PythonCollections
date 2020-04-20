import glob
import os
import sys
from os import path
from os.path import expanduser

# CLion Activator v1.2
# By congard
# https://github.com/congard
# http://congard.pp.ua/support-me
# mailto:dbcongard@gmail.com
# https://t.me/congard
# If you have an opportunity, please buy CLion: https://www.jetbrains.com/clion/buy/

home = expanduser("~")

# searching CLion folder
clionPath = glob.glob(path.join(home, ".CLion*"))
print("Found CLion folders:", *clionPath, ' ')
print("Enter folder index (by default 0): ")

# you can specify default CLion folder by passing command-line argument - folder index
if len(sys.argv) > 1:
    index = sys.argv[1]
    print(index)
else:
    index = input()
    if index == '':
        index = 0

index = int(index)

# removing evaluation key
key = glob.glob(path.join(clionPath[index], "config/eval/CLion*evaluation.key"))[0]
print("Removing " + key)
os.remove(key)

# removing line which containing evlsprt
otherXml = path.join(clionPath[index], "config/options/other.xml")
print("Clearing evlsprt in " + otherXml)
print("Removing")

# reading other.xml
with open(otherXml, 'r') as file:
    data = file.read()

# editing
data = data.split("\n")
newFile = ''
for i in range(len(data)):
    if data[i].find("evlsprt") != -1:
        print(data[i])
    else:
        newFile += data[i] + "\n"

# saving edited other.xml
with open(otherXml, "w") as file:
    file.write(newFile)

# Removing the following entries for Windows and Linux is different.
# On Linux, they are stored simply in /home, on Windows, in the registry.
# IN WINDOWS YOU SHOULD LAUNCH THIS SCRIPT AS AN ADMINISTRATOR!

# removing clion userPrefs directory
if os.name == "posix":
    print("System: Linux")
    import shutil

    clionUserPrefs = path.join(home, ".java/.userPrefs/jetbrains/clion")
    print("Removing " + clionUserPrefs)
    shutil.rmtree(clionUserPrefs)

# removing registry entries
if os.name == "nt":
    print("System: Windows")
    import winreg

    def deleteSubkey(hkey, key):
        open_key = winreg.OpenKey(hkey, key, 0, winreg.KEY_ALL_ACCESS)
        info_key = winreg.QueryInfoKey(open_key)
        for x in range(0, info_key[0]):
            sub_key = winreg.EnumKey(open_key, 0)
            try:
                winreg.DeleteKey(open_key, sub_key)
                print("Removed %s\\%s " % (key, sub_key))
            except WindowsError:
                deleteSubkey(hkey, key + "\\" + sub_key)

        winreg.DeleteKey(open_key, "")
        open_key.Close()
        print("Removed %s" % key)

    key = R"Software\JavaSoft\Prefs\jetbrains\clion"
    print("Removing " + key + " from registry")
    try:
        deleteSubkey(winreg.HKEY_CURRENT_USER, key)
    except WindowsError as e:
        print("Are you sure you ran this script as an administrator?")
        print(str(e))
    else:
        print("Registry edited")