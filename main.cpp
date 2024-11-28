#include <pybind11/embed.h>
#include <iostream>
#include <filesystem>

#define MODEL_TYPE "bert"
#define MODEL_SIZE "base"
#define DATASET "SST-2"

namespace py = pybind11;

std::string get_current_working_directory() {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        return std::string(cwd);
    } else {
        std::cerr << "Error: Unable to get current working directory" << std::endl;
        return "";
    }
}

int main()
{
    py::scoped_interpreter guard{};

    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")("../examples/");
        // sys.attr("path").attr("append")("../");

        // Now try importing the module
        // py::module_ transformers = py::module_::import("transformers.modeling_highway_bert");

        // Example usage
        // py::object model = transformers.attr("BertForSequenceClassification")();

        // Call any methods or access attributes as needed
        // std::cout << "Module imported and object created!" << std::endl;


        // Add the transformers directory to sys.path
        // sys.attr("path").attr("append")(transformers_path.c_str());

        py::list argv;
        argv.append("standalone.py");
        argv.append("--model_type");
        argv.append(MODEL_TYPE);
        argv.append("--model_name_or_path");
        argv.append("./saved_models/bert_base-SST-2-two_stage");
        argv.append("--task_name");
        argv.append(DATASET);
        argv.append("--do_eval");
        argv.append("--do_lower_case");
        argv.append("--data_dir");
        argv.append("./data/SST-2");
        argv.append("--output_dir");
        argv.append("./saved_models/bert_base-SST-2-two_stage");
        argv.append("--plot_data_dir");
        argv.append("./plotting/");
        argv.append("--max_seq_length");
        argv.append("128");
        argv.append("--early_exit_entropy");
        argv.append("0,1,0,0,0,0,0,0,0,0,0,0");
        argv.append("--eval_highway");
        argv.append("--overwrite_cache");

        sys.attr("argv") = argv;
        
        py::module_ script = py::module_::import("standalone.py");
        script.attr("main")();
    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}