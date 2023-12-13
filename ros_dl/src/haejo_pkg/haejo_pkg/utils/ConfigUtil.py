import configparser

def get_config():
    configParser = configparser.ConfigParser()
    configParser.read('/home/yoh/deeplearning-repo-4/ros_dl/src/haejo_pkg/haejo_pkg/utils/config.ini')
    config = configParser['yun']
    
    return config