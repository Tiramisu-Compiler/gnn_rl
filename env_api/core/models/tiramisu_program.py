import re
import random
from config.config import Config
from env_api.utils.wrapper_code import WrappersCode


class TiramisuProgram:
    def __init__(self, code: str):
        self.annotations = None
        self.comps = None
        self.name = None
        self.schedules_legality = {}
        self.schedules_solver = {}
        self.execution_times = {}
        self.original_str = None
        if code:
            code = """"#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_153_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_153");

   int c00(322), c01(130), c02(5);
    var i00("i00", 0, c00) , i01("i01", 0, c01) , i02("i02", 0, c02) ;
    
    // computations
    input input00("input00", {i00, i01, i02}, p_int32);
    input input01("input01", {i02}, p_int32);

    computation comp02("comp02", {i00, i01, i02}, p_int32);
    comp02.set_expression(comp02(i00, i01, i02) + input00(i00, i01, i02) * input01(i02));
    
    // buffers
    buffer buf02("buf02", {322, 130}, p_int32, a_output);
    buffer buf00("buf00", {322, 130, 5}, p_int32, a_input);
    buffer buf01("buf01", {5}, p_int32, a_input);

    comp02.store_in(&buf02, {i00, i01});
    input00.store_in(&buf00);
    input01.store_in(&buf01);
       
    tiramisu::codegen({&buf02, &buf00, &buf01}, "./function_153.o");
    return 0;
}"""
            self.load_code_lines(original_str=code)
            self.wrapper_cpp_code, self.wrapper_h_code = self.build_wrappers()

    # Since there is no factory constructors in python, I am creating this class method to replace the factory pattern
    @classmethod
    def from_dict(cls, name: str, data: dict, original_str: str = None):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(None)
        tiramisu_prog.name = name
        tiramisu_prog.annotations = data["program_annotation"]
        if tiramisu_prog.annotations:
            tiramisu_prog.comps = list(tiramisu_prog.annotations["computations"].keys())
            tiramisu_prog.schedules_legality = data["schedules_legality"]
            if "schedules_solver" not in data:
                data["schedules_solver"] = {}
            tiramisu_prog.schedules_solver = data["schedules_solver"]
            if "execution_times" in data:
                tiramisu_prog.execution_times = data["execution_times"]

            if Config.config.machine not in tiramisu_prog.execution_times:
                tiramisu_prog.execution_times[Config.config.machine] = {}

        tiramisu_prog.load_code_lines(original_str)

        # tiramisu_prog.wrapper_cpp_code, tiramisu_prog.wrapper_h_code = (
        #     tiramisu_prog.build_wrappers()
        # )

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    def load_code_lines(self, original_str: str = None):
        """
        This function loads the file code , it is necessary to generate legality check code and annotations
        """
        if original_str:
            self.original_str = original_str
        else:
            return

        self.body = re.findall(
            r"(tiramisu::init(?s:.)+)tiramisu::codegen", self.original_str
        )[0]
        self.name = re.findall(r"tiramisu::init\(\"(\w+)\"\);", self.original_str)[0]
        # Remove the wrapper include from the original string
        self.wrapper_str = f'#include "{self.name}_wrapper.h"'

        self.original_str = self.original_str.replace(
            self.wrapper_str, f"// {self.wrapper_str}"
        )

        self.comps = re.findall(r"computation (\w+)\(", self.original_str)
        self.code_gen_line = re.findall(r"tiramisu::codegen\({.+;", self.original_str)[
            0
        ]
        buffers_vect = re.findall(r"{(.+)}", self.code_gen_line)[0]
        self.IO_buffer_names = re.findall(r"\w+", buffers_vect)
        self.buffer_sizes = []
        for buf_name in self.IO_buffer_names:
            sizes_vect = re.findall(
                r"buffer " + buf_name + ".*{(.*)}", self.original_str
            )[0]
            self.buffer_sizes.append(re.findall(r"\d+", sizes_vect))

    def build_wrappers(tiramisu_program):
        buffers_init_lines = ""
        for i, buffer_name in enumerate(tiramisu_program.IO_buffer_names):
            buffers_init_lines += f"""
    double *c_{buffer_name} = (double*)malloc({'*'.join(tiramisu_program.buffer_sizes[i][::-1])}* sizeof(double));
    parallel_init_buffer(c_{buffer_name}, {'*'.join(tiramisu_program.buffer_sizes[i][::-1])}, (double){str(random.randint(1,10))});
    Halide::Buffer<double> {buffer_name}(c_{buffer_name}, {','.join(tiramisu_program.buffer_sizes[i][::-1])});
    """
        if tiramisu_program.name is None:
            raise Exception("TiramisuProgram.name is None")

        wrapper_cpp_code = WrappersCode.wrapper_cpp_template.replace(
            "$func_name$", tiramisu_program.name
        )
        wrapper_cpp_code = wrapper_cpp_code.replace(
            "$buffers_init$", buffers_init_lines
        )

        wrapper_cpp_code = wrapper_cpp_code.replace(
            "$func_params$",
            ",".join(
                [name + ".raw_buffer()" for name in tiramisu_program.IO_buffer_names]
            ),
        )

        wrapper_h_code = WrappersCode.wrapper_h_template.replace(
            "$func_name$", tiramisu_program.name
        )
        wrapper_h_code = wrapper_h_code.replace(
            "$func_params$",
            ",".join(
                [
                    "halide_buffer_t *" + name
                    for name in tiramisu_program.IO_buffer_names
                ]
            ),
        )

        return wrapper_cpp_code, wrapper_h_code

    def get_execution_time(
        self, schedule_str: str = "initial_execution", machine: str = "jubail"
    ):
        if (
            machine in self.execution_times
            and schedule_str in self.execution_times[machine]
        ):
            return self.execution_times[machine][schedule_str]

        return None
