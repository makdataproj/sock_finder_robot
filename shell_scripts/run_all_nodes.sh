#!/bin/bash

SESSION=robot

tmux new-session -d -s $SESSION

# Pane 1 - Camera
tmux send-keys -t $SESSION "cd ~ && ./run_trt_node.sh" C-m

# Split right → Pane 2
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd ~ && ./run_serial_node.sh" C-m

# Split bottom → Pane 3
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "cd ~ && ./run_ik_node.sh" C-m


tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "cd ~ && ./run_person_target.sh" C-m


tmux select-layout -t $SESSION tiled

tmux attach -t $SESSION
