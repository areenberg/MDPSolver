'''
LOADS MDPSOLVER FOR THE TESTS IN THIS FOLDER.

Every test imports it before anything else from mdpsolver:

    from loadmdpsolver import mdpsolver, finalline, quietly

The tests run the Python code of mdpsolver in Python/src, so that they can be
run from a clone without installing anything. The compiled solvermodule does
not have to be copied there; it is loaded from the first of these places that
has one:
    1. Python/src/mdpsolver, if it has been copied there;
    2. CPP_Source_Code/build, or CPP_Source_Code/build/Release on Windows,
       where the CMake build of the wiki's Contribution guidelines puts it,
       unless it is older than one of the C++ sources (which is when CMake
       would rebuild it);
    3. the installed mdpsolver package (pip install mdpsolver, or
       pip install ./Python), whose C++ may be older than the working tree's.
Only modules this Python can load are considered, so a build for another Python
version or platform in a shared folder is passed over. The module must have
every method that the Python code calls on it, or it is older than that code.
When the module is not the one in Python/src, the final line of a test says
where it came from. Copied away from the repository, as for a test of a built
wheel, the tests find no Python/src and test the installed package as a whole.

Also here: finalline(), the line a test prints when it has passed, and
quietly(), which runs a function with everything the process prints captured,
the output of the compiled module included.
'''

import ctypes
import glob
import importlib.machinery
import importlib.metadata
import importlib.util
import os
import re
import sys
import tempfile

TESTDIR = os.path.dirname(os.path.abspath(__file__))
SRCDIR = os.path.normpath(os.path.join(TESTDIR, "..", "src"))
CPPDIR = os.path.normpath(os.path.join(TESTDIR, "..", "..", "CPP_Source_Code"))
BUILDDIRS = (os.path.join(CPPDIR, "build"), os.path.join(CPPDIR, "build", "Release"))


def compiledmodules(directory):
    '''The files in a directory that this Python can load as solvermodule.'''
    paths = [os.path.join(directory, "solvermodule" + suffix)
             for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    return [path for path in paths if os.path.isfile(path)]


SOLVERMODULENOTE = None   # said in the final line when the module is not in Python/src

if not os.path.isdir(os.path.join(SRCDIR, "mdpsolver")):
    SOLVERMODULENOTE = "Python/src was not found next to the test, so the installed mdpsolver was tested as a whole."

elif not compiledmodules(os.path.join(SRCDIR, "mdpsolver")):
    sources = glob.glob(os.path.join(CPPDIR, "*.cpp")) + glob.glob(os.path.join(CPPDIR, "*.h"))
    newestsource = max(map(os.path.getmtime, sources), default=0.0)
    builds = [path for directory in BUILDDIRS for path in compiledmodules(directory)]
    currentbuilds = [path for path in builds if os.path.getmtime(path) >= newestsource]

    if currentbuilds:
        spec = importlib.util.spec_from_file_location("mdpsolver.solvermodule", currentbuilds[0])
        solvermodule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solvermodule)
        SOLVERMODULENOTE = "The compiled solvermodule was loaded from %s." % currentbuilds[0]

    else:
        # The installed package is imported as a whole, before Python/src is on
        # the path, so that whatever it does to find the libraries its module
        # links to is done. Everything but its compiled module is then forgotten.
        try:
            import mdpsolver
        except ImportError:
            sys.exit("Validation test failed at initialization: there is no compiled solvermodule for this Python in Python/src/mdpsolver or in %s%s, and mdpsolver is not installed. Build it as described in the wiki's Contribution guidelines." % (BUILDDIRS[0], " (only one older than the C++ sources)" if builds else ""))
        solvermodule = mdpsolver.solvermodule
        for name in [name for name in sys.modules if name.split(".")[0] == "mdpsolver"]:
            del sys.modules[name]

        try:
            version = importlib.metadata.version("mdpsolver")
        except importlib.metadata.PackageNotFoundError:
            version = "(version unknown)"
        SOLVERMODULENOTE = "The compiled solvermodule was that of the installed mdpsolver %s, %s, whose C++ may be older than that of the working tree." % (version, solvermodule.__file__)
        if builds:
            SOLVERMODULENOTE += " The one in %s is older than the C++ sources and was passed over." % os.path.dirname(builds[0])

    # the Python code in Python/src imports the module as mdpsolver.solvermodule
    sys.modules["mdpsolver.solvermodule"] = solvermodule

sys.path.insert(0, SRCDIR)

import mdpsolver

if os.path.isdir(os.path.join(SRCDIR, "mdpsolver")) and not os.path.samefile(
        os.path.dirname(os.path.abspath(mdpsolver.__file__)), os.path.join(SRCDIR, "mdpsolver")):
    sys.exit("Validation test failed at initialization: mdpsolver was loaded from %s rather than from Python/src." % os.path.dirname(mdpsolver.__file__))

# the compiled module must have every method that the Python code calls on it
with open(sys.modules["mdpsolver.model"].__file__) as file:
    called = sorted(set(re.findall(r"self\.mdl\.(\w+)", file.read())))
missing = [name for name in called if not hasattr(sys.modules["mdpsolver.solvermodule"].Model, name)]
if missing:
    sys.exit("Validation test failed at initialization: the compiled solvermodule has no %s, which the Python code of mdpsolver calls, so it is older than that code. Rebuild it. (%s)" % (", ".join(missing), SOLVERMODULENOTE or "It was loaded from Python/src/mdpsolver."))


def finalline(testfile):
    '''The line a test prints when it has passed: its file name, and where the compiled module came from, if not from Python/src.'''
    name = os.path.splitext(os.path.basename(testfile))[0]
    if SOLVERMODULENOTE is None:
        return "%s: All tests passed." % name
    return "%s: All tests passed. (%s)" % (name, SOLVERMODULENOTE)


try:
    LIBC = ctypes.CDLL(None)   # the C library, to flush the output of the compiled module
except (OSError, TypeError):
    LIBC = None


def flushall():
    sys.stdout.flush()
    sys.stderr.flush()
    if LIBC is not None:
        LIBC.fflush(None)


def quietly(function):
    '''Runs a function with everything the process prints captured; returns its result and the output.'''
    flushall()
    with tempfile.TemporaryFile() as capture:
        saved = (os.dup(1), os.dup(2))
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
        try:
            result = function()
        finally:
            flushall()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        capture.seek(0)
        output = capture.read().decode(errors="replace")
    return result, output
