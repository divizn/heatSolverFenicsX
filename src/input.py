import json
class Input:
    
    def __init__(self, input_directory="solid/input/", case_parameters_path="case.json", solver_parameters_path="solver.json"):
        self.input_directory = input_directory
        self.case_parameters_path = self.input_directory + case_parameters_path
        self.solver_parameters_path = self.input_directory + solver_parameters_path 
        

        with open(self.case_parameters_path, 'r') as case_parameters_file:
            self.case_parameters = json.load(case_parameters_file)

        case_parameter_keys = ["lx", "ly", "lz", "kappa", "alpha", "initial_temp", "left_bc_temp", "right_bc_temp"]
        for key in case_parameter_keys:
            if key not in self.case_parameters:
                raise Exception("Mandatory field, " + key + " not included in " + self.case_parameters_path)
        
        with open(self.solver_parameters_path, 'r') as solver_parameters_file:
            self.solver_parameters = json.load(solver_parameters_file)


        solver_parameter_keys = ["nx", "ny", "nz", "end_time", "deltaT", "poly_order", "coupled_boundary_type", "inner_loop_iterations", "write_interval"]
        for key in solver_parameter_keys:
            if key not in self.solver_parameters:
                raise Exception("Mandatory field, " + key + " not included in " + self.solver_parameters_path)

        # Optional fields
        solver_parameter_defaults = {
            "mesh_type": "box",
            "mesh_file": None,
            "mesh_name": "mesh",
            "spatial_sampler": None,
            "spatial_sampler_params": {},
        }
        for key, default in solver_parameter_defaults.items():
            self.solver_parameters.setdefault(key, default)
        