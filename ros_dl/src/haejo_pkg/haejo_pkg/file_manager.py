from haejo_pkg.utils.ConfigUtil import get_config
from haejo_pkg.utils import Logger
from botocore.client import Config
import boto3


config = get_config()
log = Logger.Logger('haejo_file_manager.log')


def s3_put_object(local_path, filename):
    try:
        # s3_client = boto3.client(
        #     's3',
        #     aws_access_key_id=config['s3_access_key'],
        #     aws_secret_access_key=config['s3_secret'],
        #     region_name='ap-northeast-2'
        # )
        s3_client = boto3.client('s3')  # local에 설정된 aws configure의 credential 사용
        s3_client.upload_file(local_path, "haejo", filename)
        
    except Exception as e:
        log.error(f" file_manager s3_put_object : {e}")
        
        return False
    
    return True


def s3_get_object(filename):
    # s3_client = boto3.client(
    #         's3',
    #         aws_access_key_id=config['s3_access_key'],
    #         aws_secret_access_key=config['s3_secret'],
    #         region_name='ap-northeast-2'
    #     )
    s3_client = boto3.client('s3')
    response = s3_client.get_object(
        Bucket="haejo",
        Key=filename
    )
    
    log.info(response)
    
    
if __name__ == '__main__':
    s3_get_object('20231127_164627.avi')