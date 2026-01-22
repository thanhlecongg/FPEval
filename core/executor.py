from abc import abstractmethod, ABC
import os
from utils import execute_command
import re



class Executor(ABC):
    """
    Abstract base class for executors that run code in a specific programming language.
    """

    def __init__(self, timeout_limit=300):
        self.timeout_limit = timeout_limit
        
    @abstractmethod
    def execute(self, env_dir: str) -> tuple[int, int, int, str]:
        """
        Executes the code in the given environment directory.

        Parameters:
        env_dir (str): The directory of the environment where the code will be executed.

        Returns:
        tuple: A tuple containing the execution result.
        """
        pass

    def apply_code(self, code: str, source_file: str):
        """
        Applies the given code to the source file between predefined start and end lines.

        Parameters:
        code (str): The code to be inserted.
        source_file (str): The path to the source file.
        """

        code_lines = open(source_file).readlines()
        print(code_lines[0])
        print(code_lines[1])
        start_idx = None
        end_idx = None
        for idx, line in enumerate(code_lines):
            if self.start_line in line:
                start_idx = idx
            if self.end_line in line:
                end_idx = idx
        assert start_idx is not None, f"Start line {self.start_line} not found"
        assert end_idx is not None, f"End line {self.end_line} not found"

        inserted_code = "".join(
            code_lines[:start_idx + 1]) + code + "\n" + "".join(
                code_lines[end_idx:])
        with open(source_file, "w") as f:
            f.write(inserted_code)


class HaskellExecutor(Executor):
    """
    Executor for running Haskell code.
    """

    def __init__(self, timeout_limit=300):
        super().__init__(timeout_limit)

        # Predefined start and end lines for inserting code
        self.start_line = "--Program start"
        self.end_line = "--Program end"

    def extract_output(self, output: str) -> tuple[int, int, int, str]:
        """
        Extracts the output from the Haskell execution result.

        Parameters:
        output (str): The output string from the Haskell execution.

        Returns:
        tuple: A tuple containing the number of tests, executed tests, errors, and the output.
        """
        
        abs_curr_dir = os.path.abspath(os.getcwd())
        if abs_curr_dir in output:
            output = output.replace(abs_curr_dir, " ")

        output = output.replace("Killed", "")
        
        print(output)
        
        ce_pattern = re.compile(
            r"app/Main\.hs:(\d+):(\d+): error:(.+?)(?=app/Main\.hs|\Z)",
            re.DOTALL)
        ce_matches = ce_pattern.findall(output)
        if ce_matches:
            return [-1, -1, -1, output]

        summary_pattern = re.compile(
                r"Cases: (\d+)\s+Tried: (\d+)\s+Errors: (\d+)\s+Failures: (\d+)")
        summary_match = summary_pattern.findall(output)
        print(summary_match)
        print("summary_match")
        if len(summary_match) == 0:
            return [-1, -5, -5, output]
        else:
            summary_match = summary_match[-1]

        n_tests, n_executed, n_errors, n_failures = summary_match
        n_tests, n_executed, n_errors, n_failures = int(n_tests), int(n_executed), int(n_errors), int(n_failures)
        assert n_tests == n_executed, "Not all tests executed"

        if n_errors + n_failures > 0:
            return -3, n_tests, n_errors + n_failures, output

        return 0, n_tests, 0, output

    def execute(self, env_dir: str) -> tuple[int, int, int, str]:
        """
        Executes the Haskell code in the given environment directory.

        Parameters:
        env_dir (str): The directory of the environment where the code will be executed.
        Folder structure should be:
        parent_directory
        ├── haskell
        │   ├── app
        │   │   └── Main.hs
        └── └──  haskell.cabal
        
        Returns:
        tuple: A tuple containing the execution result.
        """

        print("Executing Haskell code")
        curr_dir = os.getcwd()
        os.chdir(env_dir)
        output = execute_command(f"cabal run haskell", timeout=self.timeout_limit)
        os.chdir(curr_dir)
        assert os.getcwd() == curr_dir, "Current directory not restored"
        if output == "Timeout":
            return [-2, -1, -1, "Timeout"]
        return self.extract_output(output)

class OcamlExecutor(Executor):
    def __init__(self, timeout_limit=300):
        super().__init__(timeout_limit)

        # Predefined start and end lines for inserting code
        self.start_line = "(* Program start *)"
        self.end_line = "(* Program end *)"
    
    def extract_output(self, output):
        abs_curr_dir = os.path.abspath(os.getcwd())
        if abs_curr_dir in output:
            output = output.replace(abs_curr_dir, " ")
        
        failed_pattern = r'FAILED: Cases: (\d+) Tried: (\d+) Errors: (\d+) Failures: (\d+) Skip:  (\d+) Todo: (\d+) Timeouts: (\d+).'

        match = re.search(failed_pattern, output)
        if match:
            n_tests = int(match.group(1))
            n_executed = int(match.group(2))
            n_errors = int(match.group(3))
            n_failures = int(match.group(4))
            n_timeouts = int(match.group(7))
            assert n_tests == n_executed, "Not all tests executed"
            return [-3, n_tests, n_errors + n_failures + n_timeouts, output]
        
        ce_pattern = re.compile(
            r'File "([^"]+)", line (\d+), characters (\d+)-(\d+):.*\n.*\nError.*',
            re.DOTALL
        )
        
        matches = ce_pattern.findall(output)
        
        if matches:
            return [-1, -1, -1, output]
 
        # Fatal error (eg stack overflow)
        if "Fatal error" in output: 
            return [-3, -1, -1, output]
        
        ok_pattern = re.compile(r"Ran:\s(\d+)\stests.*\n(OK)")
        match = re.search(ok_pattern, output)
        if match:
            n_tests = match.group(1)
            return [0, n_tests, 0, output]
        else:
            print(output)
            assert False, "No match found"
        
    def execute(self, env_dir) -> tuple[int, int, int, str]:
        """
        Executes the Ocaml code in the given environment directory.

        Parameters:
        env_dir (str): The directory of the environment where the code will be executed.
        Folder structure should be:
        parent_directory
        ├── ocaml
        │   ├── bin
        │   │   │   dune
        └── └── └── main.ml
        
        Returns:
        tuple: A tuple containing the execution result.
        """
        
        print("Executing OCaml code")
        curr_dir = os.getcwd()
        os.chdir(env_dir)
        compile_output = execute_command(f"opam exec -- dune build --root .", timeout=self.timeout_limit)
        print(f"Compile output: {compile_output}")
        execute_output = execute_command( f"opam exec -- dune exec ocaml --root .",
            timeout=self.timeout_limit
        )
        print(f"Execute output: {execute_output}")
        os.chdir(curr_dir)
        assert os.getcwd() == curr_dir, "Current directory not restored"
        
        if compile_output == "Timeout" or execute_output == "Timeout":
            return [-2, -1, -1, "Timeout"]
    
        return self.extract_output(execute_output)
        
class ScalaExecutor(Executor):
    def __init__(self, timeout_limit=300):
        super().__init__(timeout_limit)
        
        self.start_line = "// Program start"
        self.end_line = "// Program end"
        
    def extract_output(self, output):
        abs_curr_dir = os.path.abspath(os.getcwd())
        if abs_curr_dir in output:
            output = output.replace(abs_curr_dir, " ")
        
        fail_pattern = re.compile(
            r'Failed:\s+Total\s+(\d+),\s+Failed\s+(\d+),\s+Errors\s+(\d+),\s+Passed\s+(\d+)',
            re.MULTILINE
        )
        fail_match = fail_pattern.search(output)
        if fail_match:
            total_tests = int(fail_match.group(1))
            failed_tests = int(fail_match.group(2))
            if failed_tests > 0:
                return [-3, total_tests, failed_tests, output]
        
        ok_pattern = re.compile(
            r'Passed:\s+Total\s+(\d+),\s+Failed\s+0',
            re.MULTILINE
        )
        ok_match = ok_pattern.search(output)
        if ok_match:
            total_tests = int(ok_match.group(1))
            return [0, total_tests, 0, output]
        
        assert "[error]" in output, "No error message found"
        return [-1, -1, -1, output]

    
    def execute(self, env_dir) -> tuple[int, int, int, str]:
        """
        Executes the Scala code in the given environment directory.

        Parameters:
        env_dir (str): The directory of the environment where the code will be executed.
        Folder structure should be:
        parent_directory
        ├── scala
        │   ├── main
        │   │   ├── scala
        │   │   │    └── Main.scala
        │   ├── test
        │   │   ├── scala
        └── └── └──  └── MySuite.scala
        
        Returns:
        tuple: A tuple containing the execution result.
        """
        
        print("Executing Scala code")
        curr_dir = os.getcwd()
        os.chdir(env_dir)
        print(os.getcwd())
        execute_output = execute_command(f"sbt test", timeout=self.timeout_limit)
        
        os.chdir(curr_dir)
        assert os.getcwd() == curr_dir, "Current directory not restored"
        
        if execute_output == "Timeout":
            return [-2, -1, -1, "Timeout"]
        
        return self.extract_output(execute_output)
    
class JavaExecutor(Executor):
    def __init__(self, timeout_limit=300):
        super().__init__(timeout_limit)
        self.start_line = "// Program start"
        self.end_line = "// Program end"

    def extract_output(self, output: str):
        """
        Parses the Maven test output to determine the results of the test execution.

        Parameters:
        output (str): The raw output from the Maven test command.

        Returns:
        list: A list containing the execution status, total test count, failed test count, and raw output.
        """
        abs_curr_dir = os.path.abspath(os.getcwd())
        output = output.replace(abs_curr_dir, " ")
       
        fail_pattern = re.compile(
            r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)',
            re.MULTILINE
        )
        match = fail_pattern.search(output)
        if match:
            total = int(match.group(1))
            failures = int(match.group(2)) + int(match.group(3))
            if failures > 0:
                return [-3, total, failures, output]
            return [0, total, 0, output]

        if "BUILD FAILURE" in output or "[ERROR]" in output:
            return [-1, -1, -1, output]

        assert False, "Unexpected output format"

    def execute(self, env_dir) -> tuple[int, int, int, str]:
        """
        Executes the Java code using Maven in the specified environment directory.

        Assumes the directory is structured as a Maven project:
        env_dir/
        ├── src/
        │   ├── main/
        │   │   └── java/
        │   │       └── Main.java
        │   └── test/
        │       └── java/
        │           └── MainTest.java
        └── pom.xml

        Parameters:
        env_dir (str): The root directory of the Java Maven project.

        Returns:
        tuple: A tuple containing [status_code, total_tests, failed_tests, output].
        """
        print("Executing Java code using Maven")
        curr_dir = os.getcwd()
        os.chdir(env_dir)

        try:
            output = execute_command("mvn test", timeout=self.timeout_limit)
        finally:
            os.chdir(curr_dir)
       
        if output == "Timeout":
            return [-2, -1, -1, "Timeout"]

        return self.extract_output(output)

    def apply_code(self, code: str, source_file: str):

        with open(source_file, 'r') as f:
            original_lines = f.readlines()

        raw_lines = code.splitlines()
        imports = [ln.strip() for ln in raw_lines if ln.strip().startswith(('import ', 'from '))]
        classes = [ln.strip() for ln in raw_lines ]
        i = 0
        for idx, ln in enumerate(raw_lines):
            if ln.strip().startswith('public class'):
                i = 1
                raw_lines[idx] = ''
                break

        if i == 1:
            for idx in range(len(raw_lines) - 1, -1, -1):
                if raw_lines[idx].strip() == '}':
                    raw_lines[idx] = ''  # Loại bỏ dấu '}' cuối cùng
                    break
        start_body = 0
        end_body = len(raw_lines)
        
        
        # Dedent snippet lines
        snippet = raw_lines[start_body:end_body]
        indents = [len(ln) - len(ln.lstrip()) for ln in snippet if ln.strip()]
        min_indent = min(indents) if indents else 0
        body_lines = [ln[min_indent:] for ln in snippet]

        #  Remove import lines from body
        body_lines = [ln for ln in body_lines if not ln.strip().startswith(('import ', 'from '))]
        

        # Detect and strip "class Solution { ... }"
        cleaned_body = []
        inside_solution = False
        brace_depth = 0

        for ln in body_lines:
            stripped = ln.strip()

            # Detect start of class Solution
            if not inside_solution and stripped.startswith("class Solution"):
                inside_solution = True
                # Count opening brace on same line, if exists
                if "{" in stripped:
                    brace_depth = 1
                continue

            if inside_solution:
                # Count braces
                brace_depth += ln.count("{")
                brace_depth -= ln.count("}")

                # If closing brace of Solution class reached → skip it
                if brace_depth == 0:
                    inside_solution = False
                    continue

                # Otherwise keep inner content
                cleaned_body.append(ln)
                continue

            # Outside Solution class → keep normal lines
            cleaned_body.append(ln)

        body_lines = cleaned_body
        static_fixed_body = []
        for ln in body_lines:
            stripped = ln.strip()

            if stripped.startswith("public ") and " static " not in stripped:
                # insert static after 'public'
                ln = ln.replace("public ", "public static ", 1)

            static_fixed_body.append(ln)

        body_lines = static_fixed_body


        
        print(f"🧩 Extracted body lines:")
        for ln in body_lines:
            print("   ", ln)
        pkg_idx = 0
        for idx, ln in enumerate(original_lines):
            if ln.strip().startswith('package '):
                pkg_idx = idx
                break


        # Insert imports
        
        new_lines = original_lines[:pkg_idx+1] + [imp + '\n' for imp in imports] + original_lines[pkg_idx+1:]

        start_idx = end_idx = None
        for idx, ln in enumerate(new_lines):
            if self.start_line in ln:
                start_idx = idx
            if self.end_line in ln:
                end_idx = idx
        assert start_idx is not None, f"Start marker '{self.start_line}' not found"
        assert end_idx is not None, f"End marker '{self.end_line}' not found"
        print(f"🔖 Start marker at line {start_idx}, End marker at line {end_idx}")
        print(f"New lines = {str(new_lines[:start_idx+1])}")
       
        # Build final content
        final = []
        final.extend(new_lines[:start_idx+1])
        final.append("\n")
        final.extend([ln + '\n' for ln in body_lines])
        final.append("\n")
        final.extend(new_lines[end_idx:])

        with open(source_file, 'w') as f:
            f.writelines(final)
    
