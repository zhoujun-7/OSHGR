import time
import ctypes


class DataGlove:
    def __init__(self, ):
        self.dll = ctypes.CDLL("Lib/source_data_glove/WISEGLOVEU3D.dll")
        self.cpp2py()
        self.exam()

    def exam(self, ):
        self.is_connect = self.dll.wgInit()
        self.sensor_num = self.dll.wgGetNumOfSensor()
        assert self.is_connect, 'No data glove detected.'
        assert self.sensor_num, 'Only support wiseglove14 (contains 14 sensors).'

    def cpp2py(self, ):
        self.dll.wgInit.restype = ctypes.c_bool
        self.dll.wgGetNumOfSensor.restype = ctypes.c_int
        self.dll.wgGetData.restype = ctypes.c_uint

        self.is_connect = ctypes.c_bool
        self.sensor_num = ctypes.c_int
        self.sensor_value_cpp = (ctypes.c_ushort*14)()
        self.timestamp = ctypes.c_uint

    def get_sensor_value(self, ):
        self.timestamp=self.dll.wgGetData(self.sensor_value_cpp)
        
        if self.timestamp == 0:
            time.sleep(0.01)
            return self.get_sensor_value()
        else:
            self.sensor_value = [i for i in self.sensor_value_cpp]
            return self.sensor_value


    def quit(self, ):
        self.dll.wgClose()