import sqlite3
import os

databaseName = "example.db"

#os.remove(databaseName)

conn = sqlite3.connect(databaseName)

c = conn.cursor()

# Create table
# c.execute('''CREATE TABLE fileinfoindex
#              (filefullname TEXT PRIMARY KEY,
#                filename TEXT,
#                 filesize INT
#                 )''')

# Insert a row of data
c.execute("INSERT INTO fileinfoindex VALUES ('2006-01-05','BUY', 3)")

c.execute("INSERT INTO fileinfoindex VALUES ('2006-01-051','BUY', 4)")

c.execute("INSERT INTO fileinfoindex VALUES ('2006-01-052','BUY', 4)")

c.execute("INSERT INTO fileinfoindex VALUES ('2006-01-053','BUY', 3)")

# Save (commit) the changes
conn.commit()

c.execute(""" SELECT filefullname, filesize, filename
                     FROM fileinfoindex
                     WHERE
                            filesize IN (SELECT 
                                    filesize
                                FROM
                                    fileinfoindex
                                GROUP BY filesize
                                HAVING COUNT(*) > 1) ORDER BY filesize DESC """)

print(c.fetchall())

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
