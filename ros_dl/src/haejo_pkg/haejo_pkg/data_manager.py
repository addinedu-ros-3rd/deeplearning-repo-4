from .utils import DB, Logger

db = DB.DB()
log = Logger.Logger('data_manager.py')

def insert_req(module):
    try:
        req_id = db.callProcReturn('insert_req', (module, '@requestId'))
        log.info(req_id)
        
        return req_id
        
    except Exception as e:
        log.error(f" data_manager insert_req : {e}")
        
        
def insert_res(req_id, result, file_path):
    try:
        if len(result) > 256:
            result = result[:256]
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
        
        
def select_video(module, start_date, end_date):
    try:
        sql = ("select t1.request_at, t2.name, t1.result, t1.response_at, t1.file_path "
                "from (select req.id as req_id, res.id as res_id, req.request_at, res.result, res.response_at, req.module_id, res.file_path "
		                "from response res "
		                "join request req "
		                "on res.request_id = req.id) t1 "
                "join (select md.name, md.id "
		                "from request req "
		                "join module md "
		                "on req.module_id = md.id) t2 "
                "on t1.module_id = t2.id "
                "where 1=1 "
                "AND t1.request_at IS NULL OR t1.request_at >= '" + start_date + " 00:00:00' "
                "AND t1.request_at IS NULL OR t1.request_at <= '" + end_date + " 23:59:59' ")
        
        if module != "ALL":
            sql += "AND t2.name = '" + module + "' "
            
        sql += "GROUP BY t1.res_id " + \
               "ORDER BY t1.response_at desc "
        
        db.execute(sql)
        videoList = db.fetchAll()
        
        return videoList
        
    except Exception as e:
        log.error(f" data_manager select_video : {e}")