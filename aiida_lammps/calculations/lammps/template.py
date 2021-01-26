import json
import os
from collections import Iterable
from string import Template

import numpy as np
from aiida.common import AttributeDict, CalcInfo, CodeInfo
from aiida.engine import BaseRestartWorkChain, CalcJob, ExitCode, \
    process_handler, ProcessHandlerReport, submit, while_
from aiida.orm import Code, Dict, SinglefileData, StructureData
from ase import Atoms

from aiida_lammps.common.generate_structure import generate_lammps_structure


class TemplateCalculation(CalcJob):
    _INPUT_FILE_NAME = 'input.in'
    _INPUT_STRUCTURE = 'input.data'

    _DEFAULT_OUTPUT_FILE_NAME = 'log.lammps'
    _DEFAULT_TRAJECTORY_FILE_NAME = 'trajectory.lammpstrj'
    _DEFAULT_OUTPUT_INFO_FILE_NAME = "system_info.dump"
    _DEFAULT_OUTPUT_RESTART_FILE_NAME = 'lammps.restart'

    _retrieve_list = []
    _retrieve_temporary_list = []
    _cmdline_params = ['-in', _INPUT_FILE_NAME]
    _stdout_name = None

    @classmethod
    def define(cls, spec):
        super(TemplateCalculation, cls).define(spec)
        spec.input('structure', valid_type=(StructureData, SinglefileData),
                   required=False, help='the structure')
        spec.input('template', valid_type=str, help='the input template file',
                   required=False, non_db=True)
        spec.input('variables', valid_type=dict,
                   help='input variables in template', required=False,
                   non_db=True)
        spec.input('kinds', valid_type=list, help='the sequence of elements',
                   required=False, non_db=True)
        spec.input_namespace('file', valid_type=SinglefileData, required=False,
                             dynamic=True)
        spec.input('settings', valid_type=Dict, required=False,
                   help='additional input parameters')
        spec.input('metadata.options.output_filename',
                   valid_type=str, default=cls._DEFAULT_OUTPUT_FILE_NAME)
        spec.input('metadata.options.trajectory_name',
                   valid_type=str, default=cls._DEFAULT_TRAJECTORY_FILE_NAME)
        spec.input('metadata.options.info_filename',
                   valid_type=str, default=cls._DEFAULT_OUTPUT_INFO_FILE_NAME)
        spec.input('metadata.options.restart_filename',
                   valid_type=str,
                   default=cls._DEFAULT_OUTPUT_RESTART_FILE_NAME)
        spec.input('metadata.options.withmpi', valid_type=bool, default=False)

        spec.exit_code(
            200, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(
            201, 'ERROR_NO_RETRIEVED_TEMP_FOLDER',
            message='The retrieved temporary folder data node could not be accessed.')
        spec.exit_code(
            202, 'ERROR_LOG_FILE_MISSING',
            message='the main log output file was not found')
        spec.exit_code(
            203, 'ERROR_TRAJ_FILE_MISSING',
            message='the trajectory output file was not found')
        spec.exit_code(
            204, 'ERROR_STDOUT_FILE_MISSING',
            message='the stdout output file was not found')
        spec.exit_code(
            205, 'ERROR_STDERR_FILE_MISSING',
            message='the stderr output file was not found')

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(
            300, 'ERROR_LOG_PARSING',
            message=('An error was flagged trying to parse the '
                     'main lammps output log file'))
        spec.exit_code(
            310, 'ERROR_TRAJ_PARSING',
            message=('An error was flagged trying to parse the '
                     'trajectory output file'))
        spec.exit_code(
            320, 'ERROR_INFO_PARSING',
            message=('An error was flagged trying to parse the '
                     'system info output file'))

        # Significant errors but calculation can be used to restart
        spec.exit_code(
            400, 'ERROR_LAMMPS_RUN',
            message='The main lammps output file flagged an error')

    def prepare_for_submission(self, tempfolder):

        # Setup structure
        if isinstance(self.inputs.structure, StructureData):
            structure_txt, struct_transform = generate_lammps_structure(
                self.inputs.structure, kinds=self.inputs.kinds)
        elif isinstance(self.inputs.structure, SinglefileData):
            structure_txt = self.inputs.structure.get_content()
        else:
            raise TypeError(
                'Input structure must be StructureData or SinglefileData')

        with open(self.inputs.template, 'r') as tempfile:
            temp_contents = tempfile.read()
        init_temp = Template(temp_contents)
        input_txt = init_temp.safe_substitute(**self.inputs.variables)
        if 'kinds' in self.inputs:
            kind_temp = Template(input_txt)
            kind_var = {kind: kind_index + 1 for kind_index, kind in
                        enumerate(self.inputs.kinds)}
            input_txt = kind_temp.safe_substitute(**kind_var)

        # =========================== dump to file =============================
        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, 'w') as infile:
            infile.write(structure_txt)

        # ============================ calcinfo ================================
        settings = self.inputs.settings.get_dict() \
            if 'settings' in self.inputs else {}

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = self._cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self._stdout_name
        codeinfo.join_files = True

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.stdin_name = self._INPUT_FILE_NAME
        calcinfo.stdout_name = self._stdout_name
        calcinfo.retrieve_list = self._retrieve_list + [
            self.options.output_filename]
        calcinfo.retrieve_list += settings.pop('additional_retrieve_list', [])
        calcinfo.retrieve_temporary_list = self._retrieve_temporary_list
        calcinfo.codes_info = [codeinfo]

        # =========================== local_copy_list ==========================

        if 'file' in self.inputs:
            calcinfo.local_copy_list = []
            for name, obj in self.inputs.file.items():
                calcinfo.local_copy_list.append(
                    (obj.uuid, obj.filename, f'{name}.pb'))

        return calcinfo


class TemplateWorkChain(BaseRestartWorkChain):
    _process_class = TemplateCalculation

    @classmethod
    def define(cls, spec):
        super(TemplateWorkChain, cls).define(spec)
        spec.expose_inputs(TemplateCalculation, namespace='lmp')

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(TemplateCalculation)

    def setup(self):
        super(TemplateWorkChain, self).setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(TemplateCalculation,
                                                            'lmp'))

    @process_handler(priority=500)
    def resubmit_random_gpu_error(self, calc):
        content_string = calc.outputs.retrieved.get_object_content(
            calc.get_attribute('scheduler_stdout'))

        gpu_error = "Invalid argument: No OpKernel was registered to support " \
                    "Op 'DescrptSeA' with these attrs."
        time_exceeded = "Total wall time:"
        if gpu_error in content_string:
            self.report("Inspect GPU Error")
            return ProcessHandlerReport(False, ExitCode(500))

        if (gpu_error not in content_string
                and time_exceeded not in content_string):
            self.report("Something wrong during moodel deviation")
            return ProcessHandlerReport(True, ExitCode(1))
        return None


def generate_atoms_dummy(atoms, index):
    atoms_dummy = atoms.copy()
    atoms_dummy.set_array('masses', atoms.get_masses())
    if isinstance(index, int):
        atoms_dummy[index].symbol = 'X'
    elif isinstance(index, Iterable):
        for i in index:
            atoms_dummy[i].symbol = 'X'
    return atoms_dummy


def submit_mixlmp(structure, dummy_index, steps, temp, lambda_f, dft_graphs,
                  dum_graphs):
    builder = TemplateCalculation.get_builder()
    # builder.metadata.dry_run = True
    builder.code = Code.get_from_string('lammps@metal')
    builder.settings = {'additional_retrieve_list': ['*.xyz', '*.out']}
    builder.metadata.options.queue_name = 'large'
    builder.metadata.options.resources = {'tot_num_mpiprocs': 4}
    builder.metadata.options.custom_scheduler_commands = '#BSUB -R "span[ptile=4]"'
    builder.template = os.path.abspath('template_lmp.in')
    files = {}
    for i, graph in enumerate(dft_graphs):
        files.update({f'dft_graph_{i}': SinglefileData(file=graph,
                                                       filename=f'dft_graph_{i}.pb')})
    for i, graph in enumerate(dum_graphs):
        files.update({f'dum_graph_{i}': SinglefileData(file=graph,
                                                       filename=f'dum_graph_{i}.pb')})
    builder.file = files
    builder.variables = {'TBD_STEPS': steps, 'TBD_TEMP': temp,
                         'TBD_LAMBDA_f': lambda_f,
                         'TBD_DFT': ' '.join([f + '.pb' for f in filter(
                             lambda x: x.startswith('dft'), files.keys())]),
                         'TBD_DUM': ' '.join([f + '.pb' for f in filter(
                             lambda x: x.startswith('dum'), files.keys())]),
                         'TBD_vel': np.random.randint(10000000),
                         'TBD_INPUT': TemplateCalculation._INPUT_STRUCTURE}
    builder.kinds = ['O', 'H', 'Na', 'Cl', 'X']
    if isinstance(structure, Atoms):
        atoms = structure
    elif isinstance(structure, StructureData):
        atoms = structure.get_ase()
    else:
        raise TypeError(
            "Unknown structure format, please use ase.Atoms or aiida.orm.StructureData")
    builder.structure = StructureData(
        ase=generate_atoms_dummy(atoms, dummy_index))
    submit(builder)


class BatchTemplateCalculation(TemplateCalculation):
    """
    OUTPUT TREE
    ├── 0
    │   ├── input.data
    │   ├── input.in
    │   ├── outputs...
    │   └── model_devi.out
    ├── folders same as 0...
    ├── _aiidasubmit.sh
    ├── graph_0.pb
    ├── graph_1.pb
    ├── graph_2.pb
    ├── graph_3.pb
    ├── _scheduler-stderr.txt
    └── _scheduler-stdout.txt
    """

    @classmethod
    def define(cls, spec):
        super(BatchTemplateCalculation, cls).define(spec)
        spec.input('conditions', valid_type=list, non_db=True)
        # spec.input('structures', valid_type=list, non_db=True)
        spec.input('template', valid_type=str,
                   required=False, non_db=True)
        # spec.input('variables', valid_type=dict,
        #            required=False, non_db=True)
        spec.input('kinds', valid_type=list,
                   required=False, non_db=True)
        spec.input_namespace('file', valid_type=SinglefileData,
                             required=False, dynamic=True)

    def prepare_for_submission(self, tempfolder):
        # Setup template
        with open(self.inputs.template, 'r') as tempfile:
            temp_contents = tempfile.read()
        lmp_template = Template(temp_contents)

        # # check variables
        # for variable in self.inputs.variables.values():
        #     if not isinstance(variable, list):
        #         raise TypeError('Values in variables must be list')
        # check kinds
        if 'kinds' in self.inputs:
            kind_var = {kind: kind_index + 1 for kind_index, kind in
                        enumerate(self.inputs.kinds)}
            lmp_template = Template(lmp_template.safe_substitute(**kind_var))

        for i, condition in enumerate(self.inputs.conditions):
            structure = condition.pop('structure')
            if isinstance(structure, StructureData):
                structure_txt, struct_transform = generate_lammps_structure(
                    structure, kinds=self.inputs.kinds)
            elif isinstance(structure, SinglefileData):
                structure_txt = structure.get_content()
            else:
                raise TypeError(
                    'Input structure must be StructureData or SinglefileData')
            input_txt = lmp_template.safe_substitute(**condition)

            # ========================= dump to file ===========================
            tempfolder.get_subfolder(i, create=True)
            input_filename = tempfolder.get_abs_path(
                f'{i}/{self._INPUT_FILE_NAME}')
            with open(input_filename, 'w') as infile:
                infile.write(input_txt)

            structure_filename = tempfolder.get_abs_path(
                f'{i}/{self._INPUT_STRUCTURE}')
            with open(structure_filename, 'w') as infile:
                infile.write(structure_txt)

            condition_filename = tempfolder.get_abs_path(
                f'{i}/condition.json')
            condition.update({'structure pk': structure.pk})
            with open(condition_filename, 'w') as infile:
                json.dump(condition,
                          infile, sort_keys=True, indent=2)

        # ============================ calcinfo ================================
        settings = self.inputs.settings.get_dict() \
            if 'settings' in self.inputs else {}

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = self._cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self._stdout_name
        codeinfo.join_files = True

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.prepend_text = (f'for i in $(seq 0 {i})\n'
                                 'do\n'
                                 'cd "${i}" || exit')
        calcinfo.append_text = ('cd ..\n'
                                'done')
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.stdin_name = self._INPUT_FILE_NAME
        calcinfo.stdout_name = self._stdout_name
        calcinfo.retrieve_list = self._retrieve_list + [
            self.options.output_filename]
        calcinfo.retrieve_list += settings.pop('additional_retrieve_list', [])
        calcinfo.retrieve_temporary_list = self._retrieve_temporary_list
        calcinfo.codes_info = [codeinfo]

        # =========================== local_copy_list ==========================

        if 'file' in self.inputs:
            calcinfo.local_copy_list = []
            for name, obj in self.inputs.file.items():
                calcinfo.local_copy_list.append(
                    (obj.uuid, obj.filename, f'{name}.pb'))

        return calcinfo
