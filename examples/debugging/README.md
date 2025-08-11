# Debugging Python Actors with pdb in Monarch

**Note: This API is experimental and subject to change in the very near future.**

Monarch supports `pdb` debugging for python actor meshes. To get started, simply define your python actor and insert a typical breakpoint in the relevant endpoint that you want to debug:

```
class DebugeeActor(Actor):
    @endpoint
    async def to_debug(self):
        rank = current_rank().rank
        breakpoint()
        rank = inner_function(rank)
        return rank + 1


def inner_function(rank):
    x = random.randint(0, 1000)
    return rank + x
```

**Note: If you are working in a python notebook, you must define the actor you want to debug in a separate file and import it into the notebook. Additionally, if your python actors are running on machines that are different from the machine running your main script, the actor must be defined outside of the main file inside a python module that exists on all machines.**

To access the debugger:

```
from monarch.actor import debug_client, proc_mesh
from my_actors import DebugeeActor

# Create a mesh with 4 hosts and 4 gpus per host
process_mesh = proc_mesh(hosts=4, gpus=4)

# Spawn the actor you want to debug on the mesh
debugee_mesh = process_mesh.spawn("debugee", DebugeeActor).get()

# Call the endpoint you want to debug, but do not call `.get()`, so that
# the call executes asynchronously
res = debugee_mesh.to_debug.call()

# Enter the debugger
debug_client().enter.call_one().get()

print(res.get())
```

Run the script, and in your terminal, you should see:

```
************************ MONARCH DEBUGGER ************************
Enter 'help' for a list of commands.
Enter 'list' to show all active breakpoints.

monarch_dbg>
```

Enter `list`, and you should see something like:

```
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| Actor Name   |   Rank | Coords                  | Hostname                    | Function           |   Line No. |
+==============+========+=========================+=============================+====================+============+
| debugee      |      0 | {'hosts': 0, 'gpus': 0} | your.host.com               | myactors.to_debug  |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      1 | {'hosts': 0, 'gpus': 1} | your.host.com               | myactors.to_debug  |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      2 | {'hosts': 0, 'gpus': 2} | your.host.com               | myactors.to_debug  |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      3 | {'hosts': 0, 'gpus': 3} | your.host.com               | myactors.to_debug  |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      4 | {'hosts': 1, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      5 | {'hosts': 1, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      6 | {'hosts': 1, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      7 | {'hosts': 1, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      8 | {'hosts': 2, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      9 | {'hosts': 2, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     10 | {'hosts': 2, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     11 | {'hosts': 2, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     12 | {'hosts': 3, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     13 | {'hosts': 3, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     14 | {'hosts': 3, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     15 | {'hosts': 3, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
monarch_dbg>
```

The table will show you all actors in your system that are currently stopped at a breakpoint, as well as some basic information about that breakpoint.

## Attaching to a specific actor

From the `monarch_dbg>` prompt, you can dive into a specific actor/breakpoint using the `attach` command, specifying the *name* and *rank* of the actor you want to attach to. E.g.:

```
monarch_dbg> attach debugee 13
Attached to debug session for rank 13 (your.host.com)
> /path/to/monarch/examples/debugging/debugging.py(16)to_debug()
-> rank = inner_function(rank)
(Pdb)
```

From here, you can send arbitrary pdb commands to `debugee 13`:

```
(Pdb) s
--Call--
> /path/to/monarch/examples/debugging/debugging.py(20)inner_function()
-> def inner_function(rank):
(Pdb) n
> /path/to/monarch/examples/debugging/debugging.py(21)inner_function()
-> x = random.randint(0, 1000)
(Pdb) n
> /path/to/monarch/examples/debugging/debugging.py(22)inner_function()
-> return rank + x
(Pdb) x
655
```

The debugger will automatically detach from `debugee 13` when the endpoint completes, but you can detach early using the `detach` command:

```
(Pdb) detach
Detaching from debug session for rank 13 (your.host.com)
Detached from debug session for rank 13 (your.host.com)
monarch_dbg>
```

When you `list` the breakpoints again, you can see that `debugee 13` has been updated:

```
monarch_dbg> list
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| Actor Name   |   Rank | Coords                  | Hostname                    | Function                 |   Line No. |
+==============+========+=========================+=============================+==========================+============+
| debugee      |      0 | {'hosts': 0, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      1 | {'hosts': 0, 'gpus': 1} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      2 | {'hosts': 0, 'gpus': 2} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      3 | {'hosts': 0, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      4 | {'hosts': 1, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      5 | {'hosts': 1, 'gpus': 1} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      6 | {'hosts': 1, 'gpus': 2} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      7 | {'hosts': 1, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      8 | {'hosts': 2, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      9 | {'hosts': 2, 'gpus': 1} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     10 | {'hosts': 2, 'gpus': 2} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     11 | {'hosts': 2, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     12 | {'hosts': 3, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     13 | {'hosts': 3, 'gpus': 1} | your.host.com               | my_actors.inner_function |         22 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     14 | {'hosts': 3, 'gpus': 2} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     15 | {'hosts': 3, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
```

You can reattach to `debugee 13` and use the `continue` command to tell the actor to advance indefinitely. When the `pdb` session detaches, you can then observe that there is no more breakpoint for `debugee 13`.

```
monarch_dbg> attach debugee 13
Attached to debug session for rank 13 (your.host.com)
655
(Pdb) continue
Detaching from debug session for rank 13 (your.host.com)
Detached from debug session for rank 13 (your.host.com)
monarch_dbg> list
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| Actor Name   |   Rank | Coords                  | Hostname                    | Function           |   Line No. |
+==============+========+=========================+=============================+====================+============+
| debugee      |      0 | {'hosts': 0, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      1 | {'hosts': 0, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      2 | {'hosts': 0, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      3 | {'hosts': 0, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      4 | {'hosts': 1, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      5 | {'hosts': 1, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      6 | {'hosts': 1, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      7 | {'hosts': 1, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      8 | {'hosts': 2, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |      9 | {'hosts': 2, 'gpus': 1} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     10 | {'hosts': 2, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     11 | {'hosts': 2, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     12 | {'hosts': 3, 'gpus': 0} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     14 | {'hosts': 3, 'gpus': 2} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
| debugee      |     15 | {'hosts': 3, 'gpus': 3} | your.host.com               | my_actors.to_debug |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------+------------+
```

## Casting commands to multiple actors

You can send `pdb` commands to multiple actors on the same actor mesh at once using the `cast` command. The usage of this command is:

```
monarch_dbg> cast <actor_name> ranks(<ranks>) <pdb_command>
```

There are several ways to specify ranks:
- `ranks(<rank>)`: sends a command to a single rank specified by `rank` without attaching.
- `ranks(<r1>,<r2>,<r3>)`: sends a command to the specified comma-separated list of ranks.
- `ranks(<r_start>:<r_stop>:<r_step>)`: like python list indexing syntax, each argument is optional; sends a command to the ranks in the interval `[r_start, r_stop)` with a step size of `r_step`.
- `ranks(hosts=<...>, gpus=<...>)`: sends a command to the specified set of coordinates, where `<...>` can contain a single rank, a comma-separated list of ranks, or a range of ranks. At least one dimension must be specified, but both dimensions do not need to be included.

So, the following sequence of commands will:
1. Tell rank 0 and 1 to advance to the next line.
2. Tell ranks 2, 4 and 6 to step into the function call at the current line.
3. Tell hosts 2 and 3 and gpus 1 through 3 to continue execution.
4. List the current breakpoints to show the result of our commands.

```
monarch_dbg> cast debugee ranks(0,1) n
monarch_dbg> cast debugee ranks(2:7:2) s
monarch_dbg> cast debugee ranks(hosts=2:, gpus=1:) c
monarch_dbg> list
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| Actor Name   |   Rank | Coords                  | Hostname                    | Function                 |   Line No. |
+==============+========+=========================+=============================+==========================+============+
| debugee      |      0 | {'hosts': 0, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         17 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      1 | {'hosts': 0, 'gpus': 1} | your.host.com               | my_actors.to_debug       |         17 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      2 | {'hosts': 0, 'gpus': 2} | your.host.com               | my_actors.inner_function |         20 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      3 | {'hosts': 0, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      4 | {'hosts': 1, 'gpus': 0} | your.host.com               | my_actors.inner_function |         20 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      5 | {'hosts': 1, 'gpus': 1} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      6 | {'hosts': 1, 'gpus': 2} | your.host.com               | my_actors.inner_function |         20 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      7 | {'hosts': 1, 'gpus': 3} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |      8 | {'hosts': 2, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
| debugee      |     12 | {'hosts': 3, 'gpus': 0} | your.host.com               | my_actors.to_debug       |         16 |
+--------------+--------+-------------------------+-----------------------------+--------------------------+------------+
```

## Post-mortem debugging

If an actor endpoint raises an error after a breakpoint has been hit, execution will stop where the error was raised to allow for post-mortem debugging. Currently, this is enabled by default, with no way to disable it, and no way to access post-mortem debugging unless the endpoint already hit a breakpoint -- this should all be changing very soon.

## Exiting the debugger

To safely exit the monarch debugger, from the `monarch_dbg>` prompt, simply enter `c` or `continue`. This will tell all ranks to continue execution and exit their `pdb` sessions.
