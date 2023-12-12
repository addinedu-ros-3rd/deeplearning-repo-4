from .utils import DB, Logger

db = DB.DB()
log = Logger.Logger('haejo_data_manager.log')

def insert_req(module):
    try:
        req_id = db.callProcReturn('insert_req', (module, '@requestId'))
        log.info(req_id)
        
        return req_id
        
    except Exception as e:
        log.error(f" data_manager insert_req : {e}")
        
        
def insert_res(req_id, result, file_path):
    try:
        db.callProc('insert_res', (req_id, result, file_path))
        
    except Exception as e:
        log.error(f" data_manager insert_res : {e}")
        
        
def select_module():
    try:
        sql = "SELECT name FROM module order by id"
        db.execute(sql)
        moduleList = db.fetchAll()
        
        return moduleList
        
    except Exception as e:
        log.error(f" data_manager select_module : {e}")
        
        
if __name__ == '__main__':
    log.info('data_manager ok')
    req_id = insert_req('detect_desk')
    insert_res(req_id, 'vinyl', 'python_test')