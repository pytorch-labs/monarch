#!/bin/bash
set -x

generate_random_string() {
  length=$1
  tr -dc 'a-zA-Z0-9' < /dev/urandom | fold -w "$length" | head -n 1
}

get_grandparent_process() {
  ps -p $PPID -o ppid=
}

session="monarch_interactive"
hyper="//monarch/hyper:hyper"
pingpong="//monarch/examples/ping_pong:pingpong_worker"
mode="@//mode/dev-nosan"
enable_profile=0
exec_id="$(generate_random_string 10)"
n_messages=10000
HYPERACTOR_BOOTSTRAP_ADDR='unix!/tmp/monarch.sock'


# start the session
tmux kill-window -t $session
tmux setenv -g RUST_LOG error
tmux setenv -g HYPERACTOR_EXECUTION_ID "$exec_id"
tmux setenv -g MONARCH_OTEL_LOG debug
tmux new-window  -d -k -n "$session" \; split -t "$session" -h \; splitw -t "$session" \; splitw -t "$session"
tmux setenv -g HYPERACTOR_EXECUTION_ID "$exec_id"
tmux setenv -g MONARCH_OTEL_LOG debug

# ensure everything is built
echo "building hyper $hyper"
buck2 build $mode $hyper || exit $?
echo "building example $pingpong"
buck2 build $mode $pingpong || exit $?

# run a command in the spesified tmux pane
runmux() {
    tmux send -t "$session.$1" "$2" C-m
}

# Test the function
rm /tmp/monarch.sock;

runmux 1 "buck run $mode $hyper -- serve -a '$HYPERACTOR_BOOTSTRAP_ADDR' || tmux kill-window -t $session"
runmux 2 "buck run $mode $pingpong -- -i $n_messages -p 'ping[1]' -a='$HYPERACTOR_BOOTSTRAP_ADDR'"
sleep 1
runmux 3 "buck run $mode $pingpong -- -i $n_messages -p 'ping[0]' -a='$HYPERACTOR_BOOTSTRAP_ADDR'"
if [[ $enable_profile == "1" ]]; then
  sleep 4
  runmux 4 "strobe bpf --children --pids $(get_grandparent_process) --event cpu_cycles --sample-interval 1000000"
else
  runmux 4 "ptail -f perfpipe_monarch_tracing | grep $exec_id"
fi

tmux selectw -t $session
