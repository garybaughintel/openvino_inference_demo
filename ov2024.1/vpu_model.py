from openvino.preprocess import PrePostProcessor
from openvino.runtime.utils.types import get_dtype
from openvino.runtime import Core, Type, serialize
from io import BytesIO
import numpy as np
import subprocess
import tempfile
import json
import os

def get_core():
    ie = Core()
    ie.set_property({'CACHE_DIR': 'cache'})
    ie.set_property("NPU", {
        "NPU_COMPILER_TYPE":"DRIVER",
        "PERFORMANCE_HINT": "LATENCY", 
        # "PERF_COUNT": "YES",
        "NPU_COMPILATION_MODE_PARAMS": "dpu-profiling=true dma-profiling=true sw-profiling=true",
        # "NPU_USE_ELF_COMPILER_BACKEND": "NO"
    })

    return ie

class VPUModel():
    def __init__(self, model_name, device_name,requests=1):

        self.core = get_core()

        self.model_name = model_name       
        self.device_name = device_name
        self.requests = requests

        if self.model_name.endswith("xml"):
            self.load_model(model_name, device_name)

        elif self.model_name.endswith("blob"):
            self.load_blob(model_name, device_name)

        self.create_infer_request(requests)

    def create_infer_request(self, requests):
        self.ireq = []
        for i in range(0,requests):
            self.ireq.append(self.compiled_model.create_infer_request())       
    
    def run(self,input_data) :
        for i in range(self.requests) :
            self.ireq[i].start_async(input_data)
        
        for i in range(self.requests) :
            self.ireq[i].wait()        
    
    def get_output(self) :
        output_data = []
        for i in range(self.requests) :
            output_data.append(self.ireq[i].get_output_tensor(0).data.astype(np.float32))
        return output_data
    
    def get_config(self, device_name):
        if "NPU" in device_name:
            return {
            "NPU_COMPILER_TYPE":"DRIVER",
            "PERFORMANCE_HINT": "THROUGHPUT",
            "PERF_COUNT": "YES",
            # "NPU_COMPILATION_MODE_PARAMS": "vertical-fusion=true",
            # "LOG_LEVEL": "LOG_INFO",

            # "NPU_PRINT_PROFILING":"JSON",
            # "NPU_PROFILING_OUTPUT_FILE":"profiling.json",
            # "NPU_PROFILING_VERBOSITY":"MEDIUM",
            # "NPU_DPU_GROUPS": 1,
            # "NPU_USE_ELF_COMPILER_BACKEND": "NO",
            # "DDR_HEAP_SIZE_MB": 2000,
            "NPU_COMPILATION_MODE_PARAMS": "dpu-profiling=true dma-profiling=true sw-profiling=true"
        }
        else:
            return {}
    
    def load_model(self, model_name, device_name):
        self.model = self.core.read_model(model_name)
        config = self.get_config(device_name)
        self.compiled_model = self.core.compile_model(self.model, device_name, config=config)


    def load_blob(self, model_name, device_name):
        with open(model_name, "rb") as fh:
            buf = BytesIO(fh.read())
            self.compiled_model = self.core.import_model(buf, device_name)

    def get_profiling(self):
        profiling = {}
        if "NPU" in self.device_name:
            self.ireq[0].get_profiling_info()
            prof_file = os.environ.get('NPU_PROFILING_OUTPUT_FILE', None)
            if prof_file and os.path.exists(prof_file):
                with open(prof_file, "r") as fp:
                    profiling = json.load(fp)
                os.remove(prof_file)
        return profiling