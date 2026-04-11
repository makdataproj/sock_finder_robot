#!/bin/bash

#set up camera, arm kinematic and rqt viewer nodes

SESSION=robot

tmux new-session -d -s $SESSION

# Pane 1 - Camera
tmux send-keys -t $SESSION "cd ~ && ./camera.sh" C-m

# Split right → Pane 2
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd ~ && ./arm_kin.sh" C-m

# Split bottom → Pane 3
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "cd ~ && ./rqt_view.sh" C-m

# Nice layout
tmux select-layout -t $SESSION tiled

# Attach
tmux attach -t $SESSION
