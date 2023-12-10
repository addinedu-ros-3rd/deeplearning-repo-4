from utils import DB, Logger

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
        
        
if __name__ == '__main__':
    log.info('data_manager ok')
    # insert_req('detect_desk')
    # insert_res(1, 'vinyl', 'python_test')