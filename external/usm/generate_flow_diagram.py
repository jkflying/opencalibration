#!/usr/bin/python

import sys
import argparse

NORMAL = "normal"
FALLBACK = "fallback"
REPEAT = "REPEAT"

class Transition:
    def __init__(self, start, edge_name, end, style=NORMAL):
        self.start = start
        self.edge_name = edge_name
        self.end = end
        self.style = style

class Function:
    def __init__(self, function_name, state_name):
        self.state_name = state_name
        self.function_name = function_name
        self.decisions = []
        self.fallback_transition = None

class Decision:
    def __init__(self, condition, transition):
        self.condition = condition
        self.transition = transition

def extract_first_wrapped(table_string, key, open_char="(", close_char=")"):
    try:
        start = table_string.index(key)
    except ValueError:
        return "", 0, len(key)

    depth = 0
    while depth == 0:
        start += 1

        if table_string[start] == close_char:
            return "", start, start

        if table_string[start] == open_char:
            depth += 1

    end = start + 1
    while depth != 0:
        end += 1
        if table_string[end] == open_char:
            depth += 1
        if table_string[end] == close_char:
            depth -= 1

    state_string = table_string[start:end+1]

    return state_string, start, end+1


def extract_all_wrapped(table_string, key, open_char="(", close_char=")", exclude_ranges=[]):
    states_strings = []
    end = 0
    while end < len(table_string):
        state_n, start_n, end_n = extract_first_wrapped(table_string[end:], key, open_char, close_char)
        if len(state_n) > 0:
            exclude = False
            for exclude_start, exclude_end in exclude_ranges:
                if end + start_n < exclude_end and end + end_n > exclude_start:
                    exclude = True
            if not exclude:
                states_strings.append(state_n)
            end += end_n
        else:
            end += len(key)
    return states_strings


def extract_transitions(file_string):
    choose_state_function,_,_ = extract_first_wrapped(file_string, "chooseNextState", "{", "}")
    table_string,_,_ = extract_first_wrapped(choose_state_function, "USM_TABLE")
    state_strings = extract_all_wrapped(table_string, "USM_STATE")
    error_state = table_string.split(",")[1].strip().split("::")[-1]
    transitions = []
    for state_string in state_strings:
        start_state = state_string.split(",")[1].strip().split("::")[-1]
        transition_strings = extract_all_wrapped(state_string, "USM_MAP")
        # print start_state, transition_strings
        state_transitions = []
        error_handled = False
        for transition_string in transition_strings:
            edge_name, end_state, _ = (transition_string.replace("("," ").replace(")"," ").split(","))
            new_transition = Transition(start_state,
                                        edge_name.strip().split("::")[-1],
                                        end_state.strip().split("::")[-1])
            if new_transition.edge_name == "ERROR":
                error_handled = True
            state_transitions.append(new_transition)
        if not error_handled:
            for t in state_transitions:
                if t.start == start_state and t.end == error_state:
                    t.edge_name += "\\nERROR"
                    error_handled = True
                    break
        if not error_handled:
            state_transitions.append(Transition(start_state, "ERROR", error_state, FALLBACK))
        transitions.extend(state_transitions)
    return transitions

def extract_functions(file_string):
    run_state_function_string, run_functions_start, run_function_end = extract_first_wrapped(file_string, "runCurrentState", "{", "}")
    table_string,_,_ = extract_first_wrapped(run_state_function_string, "USM_TABLE")
    run_state_strings = extract_all_wrapped(table_string, "USM_MAP")
    functions = []
    for run_state_string in run_state_strings:
        state_name, function_name, _ = (run_state_string.split(","))
        f = Function(function_name.strip().split("::")[-1], state_name.replace("(","").replace(")","").strip().split("::")[-1])

        function_f_strings = extract_all_wrapped(file_string, f.function_name, "{", "}", [[run_functions_start, run_function_end]])
        if len(function_f_strings) != 1:
            print(f"ERROR: your state names cannot be subsets of other state names: {state_name}")
            exit(1)

        f_string = function_f_strings[0]

        decision_table_string,_,_ = extract_first_wrapped(f_string, "USM_DECISION_TABLE");

        if decision_table_string:
            decision_strings = extract_all_wrapped(decision_table_string, "USM_MAKE_DECISION");

            for decision_string in decision_strings:
                condition,transition = decision_string[1:-1].split(",")
                f.decisions.append(Decision(condition.strip(), transition.strip().split("::")[-1]))

            fallback_transition = decision_table_string.split(",")[0].strip().split("::")[-1].replace(")","")
            if fallback_transition:
                f.fallback_transition = fallback_transition

        functions.append(f)

    return functions

def extract_implicit_transitions(functions):
    implicit_transitions = []
    for func in functions:
        if func.fallback_transition == REPEAT:
            implicit_transitions.append(Transition(func.state_name, REPEAT, func.state_name, FALLBACK))
    return implicit_transitions


def make_dot_file_string(transitions, functions):
    output = []
    output.append("digraph {")
    for t in transitions:
        weight = 1
        if t.style == NORMAL:
            style = "solid"
        elif t.style == FALLBACK:
            style = "dotted"
            weight = 0.1
        else:
            style = "dashed"

        start_function_candidates = [f for f in functions if f.state_name == t.start]
        name_extension = ""
        start_extension = ""
        if len(start_function_candidates) == 1:
            func = start_function_candidates[0]
            start_extension = f"\\n{func.function_name}"
            decision_candidates = [d for d in func.decisions if d.transition == t.edge_name]
            if len(decision_candidates) == 1:
                name_extension = f"\\n{decision_candidates[0].condition}"

        end_function_candidates = [f for f in functions if f.state_name == t.end]
        end_extension = ""
        if len(end_function_candidates) == 1:
            func = end_function_candidates[0]
            end_extension = f"\\n{func.function_name}"

        output.append("    \"{start}\" -> \"{end}\" [label=\"{name}\", "
                                                    "style=\"{style}\", "
                                                    "weight={weight}]".format(start=t.start + start_extension,
                                                                              end=t.end + end_extension,
                                                                              name=t.edge_name + name_extension,
                                                                              style=style,
                                                                              weight=weight))
    output.append("}")
    return "\n".join(output)


def help():
    print("Usage: generate_flow_diagram.py input_file [output_file]")


def main():
    parser = argparse.ArgumentParser(
        prog='generate_flow_diagram',
        description='Generate a .dot file from a file implementing a usm macro based transition table')

    parser.add_argument('-v', '--verify', action='store_true', help='exit with error if the output file doesn\'t match the generated dot file')
    parser.add_argument('filename')
    parser.add_argument('out_filename', nargs='?')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        cpp_string = file.read()

    transitions = extract_transitions(cpp_string)
    functions = extract_functions(cpp_string)
    transitions += extract_implicit_transitions(functions)
    dot_file_string = make_dot_file_string(transitions, functions)

    if args.out_filename:
        out_filename = args.out_filename
    else:
        out_filename = args.filename + ".dot"

    if args.verify:
        with open(out_filename, 'r') as output_file:
            previous_dot_file_string = output_file.read()

    with open(out_filename, 'w') as output_file:
        output_file.write(dot_file_string)

    if args.verify:
        if dot_file_string != previous_dot_file_string:
            exit(1)

if __name__ == "__main__":
    main()
