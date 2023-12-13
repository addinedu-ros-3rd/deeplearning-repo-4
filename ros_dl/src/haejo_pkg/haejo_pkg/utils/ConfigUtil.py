import configparser

def get_config():
    config = configparser.ConfigParser()
    config.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')
    dev = config['yun']
    
    return dev