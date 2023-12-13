from haejo_pkg.utils.ConfigUtil import get_config
from haejo_pkg.utils import Logger
import boto3


config = get_config()
log = Logger.Logger('haejo_file_manager.log')


def s3_put_object(local_path, filename):
    try:
        s3_client = boto3.client('s3')  # local에 설정된 aws configure의 credential 사용
        s3_client.upload_file(local_path, "haejo", filename)
        
    except Exception as e:
        log.error(f" file_manager s3_put_object : {e}")
        
        return False
    
    return True


def s3_get_url(filename):
    try:
        s3_client = boto3.client('s3')
        url = s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': "haejo", 'Key': filename } )
        
        return url
        
    except Exception as e:
        log.error(f" file_manager s3_put_object : {e}")