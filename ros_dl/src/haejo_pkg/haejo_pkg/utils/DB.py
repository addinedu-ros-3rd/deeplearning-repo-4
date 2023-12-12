from . import Logger

import mysql.connector
import configparser
import os

config = configparser.ConfigParser()
config.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')

print("cwd", os.getcwd())

dev = config['dev']

log = Logger.Logger('haejo_DB.log')

class DB():
    
    def __init__(self):
        try:
            host = dev['host']
            port = dev['port']
            user = dev['user']
            password = dev['password']
            database = dev['database']
            
            self.conn = mysql.connector.connect(host = host, 
                                                port = port, 
                                                user = user, 
                                                password = password, 
                                                database = database)
            self.cursor = self.conn.cursor(buffered=True)
   
        except Exception as e:
            log.error(f" DB __init__ : {e}")
            self.conn = None


    def disconnect(self):
        self.cursor.close()
        self.conn.close()
            
            
    def checkIfConnected(self):
        if not self.conn:
            raise Exception("Not connected to the database. Call connect() method first.")
            

    def execute(self, query, params=None):
        try:
            self.checkIfConnected()
            self.cursor.execute(query, params)
            self.conn.commit()

        except Exception as e:
            log.error(f" DB execute : {e}")


    def fetchOne(self):
        try:
            self.checkIfConnected()
            return self.cursor.fetchone()[0]

        except Exception as e:
            log.error(f" DB fetchOne : {e}")


    def fetchAll(self):
        try:
            self.checkIfConnected()
            return self.cursor.fetchall()

        except Exception as e:
            log.error(f" DB fetchAll : {e}")
            
    
    def callProc(self, proc_name, params):
        try:
            self.checkIfConnected()
            log.info((proc_name, params))
            self.cursor.callproc(proc_name, params)
            self.conn.commit()

        except Exception as e:
            log.error(f" DB callProc : {e}")
            
            
    def callProcReturn(self, proc_name, params):
        try:
            self.checkIfConnected()
            log.info((proc_name, params))
            self.cursor.callproc(proc_name, params)
            self.conn.commit()

            for result in self.cursor.stored_results():
                    result = result.fetchone()[0]

        except Exception as e:
            log.error(f" DB callProc : {e}")

        return result