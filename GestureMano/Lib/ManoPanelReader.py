import time
import numpy as np
from threading import Thread
from Lib.ManoPanel import Ui_Form


class ManoPanelReader(Ui_Form):
    def read_panel(self, ):
        mano_params = []
        for i in range(45):
            v = eval(f'self.spinBox_{i}').value()
            mano_params.append(v)
        return np.array(mano_params)

    def set_panel(self, ):
        for i in range(45):
            eval(f'self.spinBox_{i}').setValue(0)
    
    def __thead_run(self, func, *args):
        t = Thread(target=func, args=args)
        t.start()

    
    def __while_read_panel(self, ):
        while True:
            if self.queue2.empty():
                mano_params = self.read_panel()
                self.queue1.put(mano_params)
            else:
                self.queue2.get()
                self.set_panel()


    
    def while_read_panel(self, ):
        self.__thead_run(self.__while_read_panel)