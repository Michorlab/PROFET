#!/usr/bin/env python3

from sys import argv
import os

main_dir = os.getcwd()
if main_dir == "/content":
    main_dir = main_dir + "/GPA-source_code"

if "plot_result" in argv[1]:
    run_script ="python3 {main_dir}/scripts/util/plot_result.py {filename}".format(main_dir = main_dir,filename=argv[2])
else:
    from scripts.util.input_args import input_params
    p, _ = input_params()
        
    run_script ="python3 {main_dir}/scripts/GPA_NN/{model}.py".format(main_dir = main_dir, model=p.generative_model)
    
    for keybord_parameters in argv[1:]:
        run_script += " " + keybord_parameters

os.system(run_script)
